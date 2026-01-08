"""Coarse cross-view Transformer encoder for DirectionNet.

This encoder follows the requested pipeline:
- CNN backbone produces 1/8-resolution feature maps F1 and F2.
- Coarse cross-view Transformer blocks mix the two streams with self- and
  bidirectional cross-attention.
- Pooled tokens are projected to the decoder interface (1x1x1024) so the
  existing upsampling decoder can regress the full-resolution direction field.

The implementation uses ``tf.compat.v1.keras`` to match the rest of the
codebase.
"""
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import BatchNormalization, Conv2D
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Layer, LayerNormalization
from tensorflow.compat.v1.keras.layers import LeakyReLU


def gelu(x):
  """GELU activation (approximation) compatible with TF1 graph mode."""
  c0 = tf.constant(0.5, dtype=x.dtype)
  c1 = tf.constant(1.0, dtype=x.dtype)
  c2 = tf.constant(0.044715, dtype=x.dtype)
  sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)
  return c0 * x * (c1 + tf.tanh(sqrt_2_over_pi * (x + c2 * tf.pow(x, 3))))


class MultiHeadAttention(Layer):
  """Multi-head attention that supports explicit query, key, and value."""

  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.q_proj = Dense(embed_dim, use_bias=False)
    self.k_proj = Dense(embed_dim, use_bias=False)
    self.v_proj = Dense(embed_dim, use_bias=False)
    self.out_proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

  def _reshape_heads(self, x, batch_size, tokens):
    x = tf.reshape(x, [batch_size, tokens, self.num_heads, self.head_dim])
    return tf.transpose(x, [0, 2, 1, 3])  # [B, H, T, D]

  def call(self, q, k, v, training=False):
    batch_size = tf.shape(q)[0]
    tokens_q = tf.shape(q)[1]
    tokens_k = tf.shape(k)[1]

    q_proj = self._reshape_heads(self.q_proj(q), batch_size, tokens_q)
    k_proj = self._reshape_heads(self.k_proj(k), batch_size, tokens_k)
    v_proj = self._reshape_heads(self.v_proj(v), batch_size, tokens_k)

    attn_logits = tf.matmul(q_proj, k_proj, transpose_b=True) * self.scale
    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v_proj)  # [B, H, Tq, D]
    out = tf.transpose(out, [0, 2, 1, 3])  # [B, Tq, H, D]
    out = tf.reshape(out, [batch_size, tokens_q, self.embed_dim])
    out = self.out_proj(out)
    out = self.drop(out, training=training)
    return out


class MLP(Layer):
  def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(MLP, self).__init__(**kwargs)
    hidden = int(embed_dim * mlp_ratio)
    self.fc1 = Dense(hidden, activation=None)
    self.fc2 = Dense(embed_dim)
    self.drop = Dropout(dropout_rate)

  def call(self, x, training=False):
    x = self.fc1(x)
    x = gelu(x)
    x = self.drop(x, training=training)
    x = self.fc2(x)
    x = self.drop(x, training=training)
    return x


class CrossViewTransformerBlock(Layer):
  """Self- and cross-attention block for the two feature streams."""

  def __init__(self,
               embed_dim,
               num_heads,
               mlp_ratio=4.0,
               dropout_rate=0.0,
               **kwargs):
    super(CrossViewTransformerBlock, self).__init__(**kwargs)
    self.self_norm1 = LayerNormalization(epsilon=1e-6)
    self.self_norm2 = LayerNormalization(epsilon=1e-6)
    self.self_attn1 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
    self.self_attn2 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)

    self.cross_q_norm1 = LayerNormalization(epsilon=1e-6)
    self.cross_kv_norm1 = LayerNormalization(epsilon=1e-6)
    self.cross_q_norm2 = LayerNormalization(epsilon=1e-6)
    self.cross_kv_norm2 = LayerNormalization(epsilon=1e-6)
    self.cross_12 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
    self.cross_21 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)

    self.ffn_norm1 = LayerNormalization(epsilon=1e-6)
    self.ffn_norm2 = LayerNormalization(epsilon=1e-6)
    self.ffn1 = MLP(embed_dim, mlp_ratio, dropout_rate)
    self.ffn2 = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, tokens1, tokens2, training=False):
    # Self-attention within each view.
    tokens1 = tokens1 + self.self_attn1(
        self.self_norm1(tokens1), self.self_norm1(tokens1),
        self.self_norm1(tokens1), training=training)
    tokens2 = tokens2 + self.self_attn2(
        self.self_norm2(tokens2), self.self_norm2(tokens2),
        self.self_norm2(tokens2), training=training)

    # Bidirectional cross-attention (F1 ↔ F2).
    tokens1 = tokens1 + self.cross_12(
        self.cross_q_norm1(tokens1), self.cross_kv_norm1(tokens2),
        self.cross_kv_norm1(tokens2), training=training)
    tokens2 = tokens2 + self.cross_21(
        self.cross_q_norm2(tokens2), self.cross_kv_norm2(tokens1),
        self.cross_kv_norm2(tokens1), training=training)

    # Feed-forward networks.
    tokens1 = tokens1 + self.ffn1(self.ffn_norm1(tokens1), training=training)
    tokens2 = tokens2 + self.ffn2(self.ffn_norm2(tokens2), training=training)
    return tokens1, tokens2


class ConvBlock(Layer):
  """Helper block: Conv -> BN -> LeakyReLU."""

  def __init__(self, filters, kernel_size, strides, regularization=0.01, **kwargs):
    super(ConvBlock, self).__init__(**kwargs)
    self.conv = Conv2D(filters,
                       kernel_size,
                       strides=strides,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(regularization))
    self.bn = BatchNormalization()
    self.act = LeakyReLU()

  def call(self, x, training=False):
    x = self.conv(x)
    x = self.bn(x, training=training)
    return self.act(x)


class CoarseTransformerEncoder(keras.Model):
  """CNN + coarse cross-view Transformer encoder."""

  def __init__(self,
               embed_dim=256,
               num_layers=2,
               num_heads=4,
               mlp_ratio=3.0,
               dropout_rate=0.05,
               regularization=0.01):
    super(CoarseTransformerEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.inplanes = 1024

    # 1/8-resolution backbone shared across the two views (F1, F2).
    self.backbone = keras.Sequential([
        ConvBlock(embed_dim // 4, 7, 2, regularization=regularization),
        ConvBlock(embed_dim // 2, 3, 2, regularization=regularization),
        ConvBlock(embed_dim, 3, 2, regularization=regularization),
    ])

    self.blocks = [
        CrossViewTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]
    self.pre_norm1 = LayerNormalization(epsilon=1e-6)
    self.pre_norm2 = LayerNormalization(epsilon=1e-6)
    self.fuse = Dense(embed_dim * 2, activation=gelu)
    self.proj_to_decoder = Dense(1024)

  def _flatten_tokens(self, x):
    # x: [B, H, W, C] -> tokens [B, T, C]
    batch_size = tf.shape(x)[0]
    tokens = tf.reshape(x, [batch_size, -1, self.embed_dim])
    return tokens

  def call(self, img1, img2, training=False):
    # CNN backbone: F1, F2 at 1/8 scale.
    f1 = self.backbone(img1, training=training)
    f2 = self.backbone(img2, training=training)

    t1 = self._flatten_tokens(f1)
    t2 = self._flatten_tokens(f2)

    for blk in self.blocks:
      t1, t2 = blk(t1, t2, training=training)

    t1 = self.pre_norm1(t1)
    t2 = self.pre_norm2(t2)

    # Global pooling + projection to decoder interface.
    pooled = tf.concat([
        tf.reduce_mean(t1, axis=1),
        tf.reduce_mean(t2, axis=1),
    ], axis=-1)
    fused = self.fuse(pooled)
    y = self.proj_to_decoder(fused)
    y = y[:, tf.newaxis, tf.newaxis, :]  # [B,1,1,1024]
    return y