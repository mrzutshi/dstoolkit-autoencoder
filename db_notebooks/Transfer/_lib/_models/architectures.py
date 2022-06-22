# Databricks notebook source
class Architecture:
  def __init__(self, name, encoder, decoder, stimulus_size):#, heads=[], variational=False):
    self.name = name
    self.encoder = encoder
    self.decoder = decoder
    self.stimulus_size = stimulus_size
    #self.heads = heads
    #self.variational = variational
    
    
def get_batch_norm_block(dropout=0.2):
  return \
    [
      {
        "LayerType": "BatchNormalization",
        "LayerParameters":
          {}
      },
      {
        "LayerType": "Activation",
        "LayerParameters":
          {"activation": "relu"}
      },
      {
        "LayerType": "Dropout",
        "LayerParameters":
          {"rate": dropout}
      }
    ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurations of VGG ConvNets - Simonyan & Zisserman (2015)
# MAGIC 
# MAGIC To start the catalogue of Autoencoder Architectures in a reasonable place
# MAGIC then each of the later configurations will follow the *spirit* of the
# MAGIC configurations found in the original paper
# MAGIC 
# MAGIC 
# MAGIC <img src ='/files/tables/vgg_archs_table.PNG' width='70%'>

# COMMAND ----------

# DBTITLE 1,Basic MLP Architecture
ae_encoder_layers_config = \
    1 * [
        {
          "LayerType": "Dense",
          "LayerParameters":
            {"units": 64, "activation": "relu"}
        }
    ]

ae_decoder_layers_config = \
    1 * [
        {
          "LayerType": "Dense",
          "LayerParameters":
            {"units": 64, "activation": "relu"}
        }
    ]

mlp_basic_architecture = Architecture(name="mlp_basic", 
                                      encoder=ae_encoder_layers_config,
                                      decoder=ae_decoder_layers_config,
                                      stimulus_size=784)

# COMMAND ----------

# DBTITLE 1,Basic ConvNet Architecture - Smaller than anything in the table
vgg_basic_encoder_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Flatten",
            "LayerParameters": {}
          }
    ]

vgg_basic_decoder_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 128}
          },
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,8)}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_basic_architecture = Architecture(name="vgg_basic", 
                                      encoder=vgg_basic_encoder_layers_config,
                                      decoder=vgg_basic_decoder_layers_config,
                                      stimulus_size=(28,28,1))

# COMMAND ----------

vgg_basic_v2_encoder_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Flatten",
            "LayerParameters": {}
          }
    ]

vgg_basic_v2_decoder_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 128}
          },
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,8)}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_basic_v2_architecture = Architecture(name="vgg_basic_v2", 
                                      encoder=vgg_basic_encoder_layers_config,
                                      decoder=vgg_basic_decoder_layers_config,
                                      stimulus_size=(28,28,1))

# COMMAND ----------

vgg_basic_encoder_plus_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Flatten",
            "LayerParameters": {}
          }
    ]

vgg_basic_decoder_plus_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 128}
          },
          *get_batch_norm_block(dropout=0.1),
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,8)}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 8, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 16, "kernel_size": (3, 3), "padding": "valid"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_basic_plus_architecture = Architecture(name="vgg_basic_plus", 
                                      encoder=vgg_basic_encoder_plus_layers_config,
                                      decoder=vgg_basic_decoder_plus_layers_config,
                                      stimulus_size=(28,28,1))

# COMMAND ----------

vgg_basic_encoder_plusplus_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 32, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "GlobalMaxPooling2D",
            "LayerParameters": {}
          }
    ] + \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units":16}
          },
          *get_batch_norm_block(dropout=0.1)
    ]

vgg_basic_decoder_plusplus_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 128}
          },
          *get_batch_norm_block(dropout=0.1),
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,8)}
          }
    ] + \
    2 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "padding": "same"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 32, "kernel_size": (3, 3), "padding": "valid"}
          },
          *get_batch_norm_block(dropout=0.2),
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_basic_plusplus_architecture = Architecture(name="vgg_basic_plusplus", 
                                      encoder=vgg_basic_encoder_plusplus_layers_config,
                                      decoder=vgg_basic_decoder_plusplus_layers_config,
                                      stimulus_size=(28,28,1))

# COMMAND ----------

# DBTITLE 1,Small ConvNet Architecture - First column of the Table
vgg_A_11_encoder_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Flatten",
            "LayerParameters": {}
          }
    ]

vgg_A_11_decoder_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 4096}
          },
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,256)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_A_11_architecture = Architecture(name="vgg_A_11", 
                                     encoder=vgg_A_11_encoder_layers_config,
                                     decoder=vgg_A_11_decoder_layers_config,
                                     stimulus_size=(28,28,1))

# COMMAND ----------

# DBTITLE 1,Large ConvNet Architecture - Last column of the Table
vgg_E_19_encoder_layers_config = \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "MaxPooling2D",
            "LayerParameters":
              {"pool_size": (2, 2), "padding": "same"}
          }
    ] + \
    1 * [
          {
            "LayerType": "Flatten",
            "LayerParameters": {}
          }
    ]

vgg_E_19_decoder_layers_config = \
    1 * [
          {
            "LayerType": "Dense",
            "LayerParameters": {"units": 4096}
          },
          {
            "LayerType": "Reshape",
            "LayerParameters": {"target_shape": (4,4,256)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 512, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 256, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 128, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"}
          },
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"}
          },
          {
            "LayerType": "UpSampling2D",
            "LayerParameters":
              {"size": (2, 2)}
          }
    ] + \
    1 * [
          {
            "LayerType": "Conv2D",
            "LayerParameters":
              {"filters": 1, "kernel_size": (3, 3), "activation": "sigmoid", "padding": "same"}
          }
    ]

vgg_A_19_architecture = Architecture(name="vgg_E_19", 
                                     encoder=vgg_E_19_encoder_layers_config,
                                     decoder=vgg_E_19_decoder_layers_config,
                                     stimulus_size=(56,56,1))
