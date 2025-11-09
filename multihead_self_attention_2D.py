import tensorflow as tf
from keras.layers import Layer, Dropout, Dense, Softmax


class MultiHeadSelfAttention(Layer):
    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = None,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim if projection_dim else embedding_dim // num_heads
        self.qkv_bias = qkv_bias
        self.attention_drop_rate = attention_drop
        self.linear_drop_rate = linear_drop
        self.scale = self.projection_dim**-0.5

        self.qkv = Dense(3 * self.num_heads * self.projection_dim, use_bias=qkv_bias)
        self.proj = Dense(embedding_dim, use_bias=qkv_bias)
        self.attn_dropout = Dropout(attention_drop)
        self.linear_dropout = Dropout(linear_drop)
        self.softmax = Softmax()

    def build(self, input_shape):
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, projection_dim)"""
        # Reshape: (batch_size, seq_len, num_heads, projection_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # Transpose to: (batch_size, num_heads, seq_len, projection_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=False):
        # Get batch size from shape
        batch_size = tf.shape(x)[0]

        # Project and reshape to (batch_size, seq_len, 3, num_heads, projection_dim)
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (batch_size, -1, 3, self.num_heads, self.projection_dim))
        
        # Transpose to (batch_size, 3, seq_len, num_heads, projection_dim)
        qkv = tf.transpose(qkv, perm=[0, 2, 1, 3, 4])
        
        # Split into q, k, v
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Split heads for each
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scale queries
        q *= self.scale

        # Attention mechanism
        # attn_logits shape: (batch_size, num_heads, seq_len, seq_len)
        attn_logits = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = self.softmax(attn_logits)
        attn = self.attn_dropout(attn, training=training)

        # Apply attention to values
        weighted_avg = tf.matmul(attn, v)
        
        # Transpose back: (batch_size, seq_len, num_heads, projection_dim)
        weighted_avg = tf.transpose(weighted_avg, perm=[0, 2, 1, 3])
        
        # Concatenate heads: (batch_size, seq_len, num_heads * projection_dim)
        weighted_avg = tf.reshape(weighted_avg, (batch_size, -1, self.num_heads * self.projection_dim))

        # Output projection
        output = self.proj(weighted_avg)
        output = self.linear_dropout(output, training=training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "projection_dim": self.projection_dim,
                "qkv_bias": self.qkv_bias,
                "attention_drop": self.attention_drop_rate,
                "linear_drop": self.linear_drop_rate,
            }
        )
        return config