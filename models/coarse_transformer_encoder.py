"""Coarse cross-view Transformer encoder for directional learning.

This encoder follows the requested pipeline:
- A shared CNN backbone produces 1/8-resolution feature maps F1 and F2.
- Coarse cross-view Transformer blocks mix the two streams with self- and
  bidirectional cross-attention (with 2D positional encodings).
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
    # [B, T, C] -> [B, H, T, D]
    x = tf.reshape(x, [batch_size, tokens, self.num_heads, self.head_dim])
    return tf.transpose(x, [0, 2, 1, 3])

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
  """Simple Transformer MLP block."""

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
  """Self- and bidirectional cross-attention block for the two feature streams."""

  def __init__(self,
               embed_dim,
               num_heads,
               mlp_ratio=4.0,
               dropout_rate=0.0,
               **kwargs):
    super(CrossViewTransformerBlock, self).__init__(**kwargs)

    # Self-attention
    self.self_norm1 = LayerNormalization(epsilon=1e-6)
    self.self_norm2 = LayerNormalization(epsilon=1e-6)
    self.self_attn1 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
    self.self_attn2 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)

    # Cross-attention: 1 <- 2 and 2 <- 1
    self.cross_q_norm1 = LayerNormalization(epsilon=1e-6)
    self.cross_kv_norm1 = LayerNormalization(epsilon=1e-6)
    self.cross_q_norm2 = LayerNormalization(epsilon=1e-6)
    self.cross_kv_norm2 = LayerNormalization(epsilon=1e-6)
    self.cross_12 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
    self.cross_21 = MultiHeadAttention(embed_dim, num_heads, dropout_rate)

    # Feed-forward networks
    self.ffn_norm1 = LayerNormalization(epsilon=1e-6)
    self.ffn_norm2 = LayerNormalization(epsilon=1e-6)
    self.ffn1 = MLP(embed_dim, mlp_ratio, dropout_rate)
    self.ffn2 = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, tokens1, tokens2, training=False):
    # Self-attention within each view.
    t1 = self.self_norm1(tokens1)
    t2 = self.self_norm2(tokens2)

    tokens1 = tokens1 + self.self_attn1(t1, t1, t1, training=training)
    tokens2 = tokens2 + self.self_attn2(t2, t2, t2, training=training)

    # Cache "old" views for symmetric cross-attention.
    t1_old = tokens1
    t2_old = tokens2

    q1 = self.cross_q_norm1(t1_old)
    k2 = self.cross_kv_norm1(t2_old)
    v2 = k2
    q2 = self.cross_q_norm2(t2_old)
    k1 = self.cross_kv_norm2(t1_old)
    v1 = k1

    # Bidirectional cross-attention (F1 <-> F2) using the same old states.
    tokens1 = tokens1 + self.cross_12(q1, k2, v2, training=training)
    tokens2 = tokens2 + self.cross_21(q2, k1, v1, training=training)

    # Feed-forward networks.
    f1 = self.ffn_norm1(tokens1)
    f2 = self.ffn_norm2(tokens2)
    tokens1 = tokens1 + self.ffn1(f1, training=training)
    tokens2 = tokens2 + self.ffn2(f2, training=training)

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


def build_2d_sincos_position_embedding(height, width, dim):
  """Builds a dynamic 2D sin/cos positional embedding [1, H*W, dim].
  
  Works with dynamic TF1 graph shapes and guarantees the output matches the 
  token count of a flattened feature map shaped [B, H*W, dim].
  """

  assert dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin/cos."

  dim_half = dim // 2
  dim_y = dim_half // 2
  dim_x = dim_half // 2

  # ---------------------------------------------------------------------------
  # Create coordinate grid dynamically (TF1 graph-compatible)
  # ---------------------------------------------------------------------------

  # y-coordinates: shape [H, 1]
  y_coords = tf.cast(tf.reshape(tf.range(height), [-1, 1]), tf.float32)

  # x-coordinates: shape [1, W]
  x_coords = tf.cast(tf.reshape(tf.range(width), [1, -1]), tf.float32)

  # Repeat to build full 2D coordinate list
  yy = tf.reshape(tf.tile(y_coords, [1, width]), [-1])  # [H*W]
  xx = tf.reshape(tf.tile(x_coords, [height, 1]), [-1]) # [H*W]

  # ---------------------------------------------------------------------------
  # 1D sin/cos helper
  # ---------------------------------------------------------------------------
  def _sincos(pos, dim_coord):
    # pos: [N]
    pos = tf.expand_dims(pos, -1)   # [N,1]

    i = tf.range(dim_coord, dtype=tf.float32)       # [dim_coord]
    freq = 1.0 / tf.pow(
        10000.0,
        2 * (i // 2) / tf.cast(dim_coord, tf.float32)
    )                                               # [dim_coord]

    angles = pos * freq                             # [N, dim_coord]

    sin = tf.sin(angles[:, 0::2])                   # [N, dim_coord/2]
    cos = tf.cos(angles[:, 1::2])                   # [N, dim_coord/2]

    return tf.concat([sin, cos], axis=-1)           # [N, dim_coord]

  # ---------------------------------------------------------------------------
  # Encode Y and X axes separately
  # ---------------------------------------------------------------------------
  emb_y = _sincos(yy, dim_y)  # [H*W, dim_y]
  emb_x = _sincos(xx, dim_x)  # [H*W, dim_x]

  # Concatenate → [N, dim]
  pos = tf.concat([emb_y, emb_x], axis=-1)

  # Reshape → [1, H*W, dim]  (batch dimension of 1)
  pos = tf.reshape(pos, [1, -1, dim])

  return pos
ape(pos, [1, -1, dim])       # [1, H*W, dim]
  return pos


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

    # Coarse cross-view Transformer blocks.
    self.blocks = [
        CrossViewTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]

    self.pre_norm1 = LayerNormalization(epsilon=1e-6)
    self.pre_norm2 = LayerNormalization(epsilon=1e-6)

    # Projection to decoder bottleneck.
    self.fuse = Dense(embed_dim * 2, activation=gelu)
    self.proj_to_decoder = Dense(1024)

  def _flatten_tokens(self, x):
    # x: [B, H, W, C] -> tokens [B, T, C]
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, [batch_size, -1, self.embed_dim])

  def call(self, img1, img2, training=False):
    # CNN backbone: F1, F2 at 1/8 scale.
    f1 = self.backbone(img1, training=training)  # [B, H/8, W/8, C]
    f2 = self.backbone(img2, training=training)  # [B, H/8, W/8, C]

    b = tf.shape(f1)[0]
    h = tf.shape(f1)[1]
    w = tf.shape(f1)[2]

    t1 = self._flatten_tokens(f1)  # [B, T, C]
    t2 = self._flatten_tokens(f2)  # [B, T, C]

    # 2D sinusoidal positional encodings (shared across the two views).
    pos = build_2d_sincos_position_embedding(h, w, self.embed_dim)  # [1, T, C]
    t1 = t1 + pos
    t2 = t2 + pos

    # Cross-view Transformer blocks.
    for blk in self.blocks:
      t1, t2 = blk(t1, t2, training=training)

    t1 = self.pre_norm1(t1)
    t2 = self.pre_norm2(t2)

    # Global pooling + projection to decoder interface.
    pooled1 = tf.reduce_mean(t1, axis=1)  # [B, C]
    pooled2 = tf.reduce_mean(t2, axis=1)  # [B, C]
    pooled = tf.concat([pooled1, pooled2], axis=-1)  # [B, 2C]

    fused = self.fuse(pooled)             # [B, 2C]
    y = self.proj_to_decoder(fused)       # [B, 1024]
    y = y[:, tf.newaxis, tf.newaxis, :]   # [B, 1, 1, 1024]
    return y
