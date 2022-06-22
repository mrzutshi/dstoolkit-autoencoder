# Databricks notebook source
RUN_TESTS = False

# COMMAND ----------

# DBTITLE 1,Run the utils notebook 
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./architectures

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base Autoencoder Class
# MAGIC This the main parent class for the autoencoder which will be used as a base class for VAE and AE model classes 
# MAGIC 
# MAGIC ### Class attributes
# MAGIC - model_name: Name of the model
# MAGIC - encoder_layers_config: A list of dictionaries that defines the architecture for encoder
# MAGIC - decoder_layers_config: A list of dictionaries that defines the architecture for decoder
# MAGIC - original_dim: The input dimension for the NN
# MAGIC - latent_dim: The dimension for latent space
# MAGIC - hvd_flag: Boolean flag to enable horovod distribution
# MAGIC 
# MAGIC ### Class methods
# MAGIC - fit: This method is derived from the keras model fit and uses either horovod ft or the regular fit ased on the hvd_flag
# MAGIC - predict: This method is derived from the keras model predict
# MAGIC - summary: This method is derived from the keras model summary. It prints out the model architecture
# MAGIC - compile: This method is derived from the keras model compile. It gives an option to compile it using the hvd distributor
# MAGIC - create_checkpoint_cb: To create callback for model checkpoints to be used in fit method
# MAGIC - load_weights: Load model weights from checkpoint directory
# MAGIC - latest_checkpoint: Load the latest model checkpoint
# MAGIC - save_model: Save the model to a directory
# MAGIC - load_model: Load the model from a directory
# MAGIC - get_tensorflow_callback: Returns tensorboard callbacks
# MAGIC 
# MAGIC ### Internal methods
# MAGIC - init_hvd: To initialize horovod distribution framework
# MAGIC - emit_config: TBD
# MAGIC - compile_hvd: Compile method to be used for hvd compile and called in the class compile method
# MAGIC - fit_hvd: Fit method for hvd distribution used in the class fit method

# COMMAND ----------

# DBTITLE 1,Base Autoencoder Class
class AutoEncoder(NnUtils):
  def __init__(self, architecture, experiment_name, latent_dim, hvd_flag=False, root_dir="/dbfs/FileStore/jf-cache/models"):
    
    self.encoder_layers_config = architecture.encoder
    self.decoder_layers_config = architecture.decoder
    
    self.experiment_name = experiment_name
    self.model_name = NnUtils._postfix(architecture.name)
    
    self.original_dim = architecture.stimulus_size
    self.latent_dim = latent_dim
    
    self.hvd_flag = hvd_flag
    
    self.root_dir = f"{root_dir}/{self.experiment_name}/{self.model_name}"
    
    os.makedirs(self.root_dir, exist_ok=True)
    os.makedirs(f"{self.root_dir}/candidates", exist_ok=True)
    
    if self.hvd_flag:
      self.__init_hvd__()
      
  def _emit_config(self):
    
    args = (self.model_name, self.encoder_layers_config, self.decoder_layers_config, self.hvd_flag)
    kwargs = {"original_dim": self.original_dim,
              "latent_dim": self.latent_dim}
    
    return args, kwargs
  
  def __init_hvd__(self):
    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    # These steps are skipped on a CPU cluster
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
      try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
      except Exception as e:
        print("Memory Growth Already Configured")
    if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
  
  def compile(self, optimizer='adam', loss=None, metrics=[]):
    if self.hvd_flag:
      self._compile_hvd(loss=loss, metrics=metrics)
    else:
      self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
      
    # Create Checkpoint Directory Ready For Model Fit
    self.create_checkpoint_cb()
  
  def _compile_hvd(self, loss=None, metrics=[]):
    # Adjust learning rate based on number of GPUs
    optimizer = keras.optimizers.Adam(lr=.01 * hvd.size())
    # Use the Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)
    self.model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)
      
  def __repr__(self):
    #self.model.summary()
    pass
    
  def fit(self, *args, **kwargs):
    if self.hvd_flag:
      # Pass epochs to this function (its inside kwargs)
      self._fit_hvd(epochs=1)
    else:
      self.model.fit(*args, **kwargs)
    
  def _fit_hvd(self, epochs=1):
    
    args, kwargs = self._emit_config()
    
    this_class = type(self).__name__
    
    image = type(self.original_dim) is tuple
    
    def train():
      mnist_data = get_data_mnist(flat=(not image))
      if this_class == "AE":
        model_obj = AE(*args, **kwargs)
      elif this_class == "VAE":
        model_obj = VAE(*args, **kwargs)
      else:
        return False
      
      model_obj.model.fit(x=mnist_data["images"]["train"], 
                          y=mnist_data["images"]["train"], 
                          epochs=epochs, 
                          validation_data=(mnist_data["images"]["test"], mnist_data["images"]["test"]),
                          callbacks = [model_obj.cp_callback])
      
    hr = HorovodRunner(np=2)
    hr.run(train)
    
    # Sort this so it gets the latest weights
    self.model.load_weights(os.path.dirname(f'{self.root_dir}/{self.model_name}/training_1/cp.ckpt'))
    
  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)
    
  def summary(self):
    return self.model.summary()
  
  def create_checkpoint_cb(self):
    
    self.ckpt_file_name = f"{self.root_dir}/training_1/cp.ckpt"
    
    os.makedirs(self.ckpt_file_name, exist_ok=True)
    
    self.checkpoint_dir = os.path.dirname(self.ckpt_file_name)
    
    # Create a callback that saves the model's weights
    # TD - Add in the call backs for saving the CVSlogger
    self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir,
                                                          save_weights_only=True,
                                                          verbose=1)
    
    ## CODE FOR LATER - KEEP
    #callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', 
    #                                              mode='min', save_weights_only=True,
    #                                              save_best_only=True),
    #          tf.keras.callbacks.CSVLogger(tmp_path, separator=',', append=True)]
  
  def load_weights(self, *args, **kwargs):
    self.model = self.model.load_weights(*args, **kwargs)
    
  def save_weights(self, *args, **kwargs):
    self.model.save_weights(*args, **kwargs)
  
  def latest_checkpoint(self,*args, **kwargs):
    return tf.train.latest_checkpoint(*args, **kwargs)
  
  def save_model(self):
    
    print(f"SAVING: {self.model_name}")
    
    args, kwargs = self._emit_config()
    with open(f"{self.root_dir}/model_config.pkl", 'wb') as file_handle:
      pickle.dump({"args": args, "kwargs": kwargs}, file_handle)
    
    self.save_weights(f"{self.root_dir}/latest_weights")
    
    #os.makedirs(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/", exist_ok=True)
    
    #self.model.save(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/ae-model")
    
  def load_model(self):
      
    if self.to_load:
      with open(f"{self.root_dir}/model_config.pkl", 'rb') as file_handle:
        args_kwargs = pickle.load(file_handle)
      
      self.__init__(*args_kwargs['args'], **args_kwargs['kwargs'])
      
      self.load_weights(f"{self.root_dir}/latest_weights")
      
      self.to_load = False
    else:
      self.to_load = True
    

    #print(f"LOADING: {self.model_name}")
    
    #with open(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/class_{self.model_type}.pkl", "rb") as file_handle:
    #  load_class_dict = pickle.load(file_handle)
      
    #class_dict = self.__dict__
    
    #for k, v in load_class_dict.items():
    #  class_dict[k] = v
      
    #self.__dict__ = class_dict
    
    #self.encoder = tf.keras.models.load_model(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/encoder-model")
    #self.decoder = tf.keras.models.load_model(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/decoder-model")
    #self.vae = tf.keras.models.load_model(f"/dbfs/FileStore/gb-cache/models/{self.model_name}/ae-model")

  def get_tensorflow_callback(self, experiment_log_dir):
    
    os.makedirs(experiment_log_dir, exist_ok=True)

    run_log_dir = f"{experiment_log_dir}/{self.model_name}__{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architectures
# MAGIC 
# MAGIC ### VGG Architectures
# MAGIC 
# MAGIC Visual Geometry Group
# MAGIC 
# MAGIC [Oxford Group](https://www.robots.ox.ac.uk/~vgg/research/very_deep/)
# MAGIC 
# MAGIC [Publication](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/simonyan15.pdf)
# MAGIC 
# MAGIC ### MLP Architectures
# MAGIC 
# MAGIC Multilayer Perceptron
# MAGIC 
# MAGIC [wiki - MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encoder and Decoders
# MAGIC 
# MAGIC Encoders make up the input (encoder) and output (decoder) modules of an autoecoder. The encoder is designed to reduce the dimensionality of the the network input through progressive layers to end provide a compressed representation of the original input to the latent space.
# MAGIC 
# MAGIC Decoders do the opposite operation, expanding the dimensionality of the activity of the latent space so that through progressive layers of the decoder the output of the final layer has the same dimensions as the original input.
# MAGIC 
# MAGIC Through training, the reconstruction loss drives the weights of the network to minimise the difference between the input to the encoder and the output of the decoder.
# MAGIC 
# MAGIC ## Links
# MAGIC 
# MAGIC [wiki - Autoencoders](https://en.wikipedia.org/wiki/Autoencoder)

# COMMAND ----------

# DBTITLE 1,Encoder Modules
class _encoder(NnUtils):#, tkm.Model):
  def __init__(self, input_dim, latent_dim, layers_config, name, heads=1, build_inputs=True):
    
    super().__init__()
    
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.layers_config = layers_config
    self.heads = heads
    
    self.name = name
    
    if build_inputs:
      self.input = [self.create_basic_inputs(self.input_dim, f"{name}_inputs") for i in range(self.heads)]
      self.penultimate = self.up_to_penultimate()
    else:
      self.input = None
      self.penultimate = None
      
    def predict(self, *args, **kwargs):
      return self.model.predict(*args, **kwargs)

  def up_to_penultimate(self):
    if len(self.input) > 1:
      return tkl.Concatenate(axis=-1)([ self.get_layers(input_, self.layers_config) for input_ in self.input])
    else:
      return self.get_layers(self.input[0], self.layers_config)
      
    
  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)
    
    
class ae_encoder(_encoder):
  def __init__(self, *args, **kwargs):
#     s_kwargs = kwargs.copy()
#     for k in ['last_activation']:#, 'build_model']:
#       if k in s_kwargs:
#         del s_kwargs[k]
#     super().__init__(*args, **s_kwargs)
    
#     #print(kwargs)
    
#     #if kwargs['build_model']:
#     kwargs = {}
#     self.__build__(**kwargs)
    self.build_arguments = []
    
    build_kwargs, init_kwargs = self.split_kwargs(kwargs, self.build_arguments)
    
    super().__init__(*args, **init_kwargs)
    self.__build__(**build_kwargs)
    
  def __build__(self):
    
    self.output = tkl.Dense(self.latent_dim, activation='linear')(self.penultimate)
    
    self.model = tkm.Model(self.input, self.output)
    
  def __call__(self, x):
    
    return self.model(x)
    

class vae_encoder(ae_encoder):
  def __init__(self, *args, **kwargs):
#     s_kwargs = kwargs.copy()
#     for k in ['last_activation']:#, 'build_model']:
#       if k in s_kwargs:
#         del s_kwargs[k]
#     super().__init__(*args, **s_kwargs)
    
#     #if kwargs['build_model']:
#     kwargs = {}
#     self.__build__(**kwargs)
    self.build_arguments = []
    
    build_kwargs, init_kwargs = self.split_kwargs(kwargs, self.build_arguments)
    
    super().__init__(*args, **init_kwargs)
    self.__build__(**build_kwargs)
  
  def __build__(self):
    
    self.z_mean, self.z_log_sigma = self.variational_layers(self.penultimate)
    
    self.output = tkl.Lambda(self.sampling)([self.z_mean, self.z_log_sigma])
    
    self.model = tkm.Model(self.input, self.output)    
  
  def __call__(self, x):
    
    return self.model(x)

  def variational_layers(self, h):
    
    return tkl.Dense(self.latent_dim)(h), tkl.Dense(self.latent_dim)(h)

  def sampling(self, args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                              mean=0.0, 
                              stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon
  
  def get_variational_loss(self, inputs, outputs):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    
    if len(reconstruction_loss.shape) > 1:
      reconstruction_loss = K.sum(K.sum(reconstruction_loss, axis=-1), axis=-1)
      
    #print(reconstruction_loss)
    
    kl_loss = 1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    #print(kl_loss)
    
    return K.mean(reconstruction_loss + kl_loss)
    

# COMMAND ----------

# DBTITLE 1,Decoder Modules
class _decoder(NnUtils):
  def __init__(self, input_dim, output_dim, layers_config, name, build_inputs=True):
    
    super().__init__()
    
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layers_config = layers_config
    
    self.name = name
    
    if build_inputs:
      self.input = self.create_basic_inputs(self.input_dim, f"{name}_input")
    else:
      self.input = None
      
  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)


class ae_decoder(_decoder):
  def __init__(self, *args, **kwargs):
    
    self.build_arguments = ['last_activation']
    
    build_kwargs, init_kwargs = self.split_kwargs(kwargs, self.build_arguments)
    
    super().__init__(*args, **init_kwargs)
    self.__build__(**build_kwargs)
    
  def __build__(self, last_activation='linear'):
    x = self.get_layers(self.input, self.layers_config)
    
    if type(self.output_dim) is tuple:
      ### TD - HACK
      self.output = x
    else:
      self.output = tkl.Dense(self.output_dim, activation=last_activation)(x)
    
    self.model = tkm.Model(self.input, self.output)    
    
  def __call__(self, x):
    
    return self.model(x)

# COMMAND ----------

# DBTITLE 1,Autoencoder Child Classes - For Each Type of Autoencoder
class AE(AutoEncoder):
  """
  The final child class of Autoencoders
  AE inherits from AutoEncoder inherits from NnUtils

  Executes the final build of the Autoencoder class by combining Encoder and Decoder Modules
  
  Features:
  - Accepts an Arcitecture class to specifiy the configuration for the Encoder and Decoder modules
  - Can have many heads (and tails) to process many heatmaps for the same play
    - This works by having an input of the form:
      Example:
        n_samples = n
        size of image = l by m
        depth of image = 1 (one is the only ption for the current implementation)
        This list would serve as input for a netowkr with 4 heads
        [ np.ndarray(shape=(n,l,m,1)), np.ndarray(shape=(n,l,m,1)), np.ndarray(shape=(n,l,m,1)), np.ndarray(shape=(n,l,m,1))]

  ...

  Attributes
  ----------
   - head_names: The names of the heatmaps that are used as input for the network
   - head_count: The number of heatmaps per play (per sample)
   - 

  Methods
  -------
   - dense_reshape - Connects a layer to a dense fully connected linear layer and then reshapes to the desired output dims
   - _postfix - Adds a short random string to a string - Used to prevent name conflicts within a network
   - split_kwargs - Splits a dictionary (kwargs) in to two dictionaries according to if the key is present in a list (args_list) 
   - get_layers - Creates a series of connected layers according to the dictionary layers_config
   - create_basic_inputs - Creates an Input layer with shape: input_dim and name: name with a post-fix
  """
  def __init__(self, architecture, experiment_name, heads, latent_dim, hvd_flag=False, last_activation='linear', variational=False):
    super().__init__(architecture, experiment_name, latent_dim, hvd_flag=hvd_flag)
    
    self.head_names = heads
    self.head_count = len(heads)
    
    # TD - GB
    if variational:
      encoder_, decoder_ = self.get_modules_config("vae")
    else:
      encoder_, decoder_ = self.get_modules_config("ae")
    
    self.encoder = encoder_['class'](self.original_dim, self.latent_dim, self.encoder_layers_config, encoder_['name'], 
                                     heads=self.head_count)
    self.decoder = [ decoder_['class'](self.latent_dim, self.original_dim, self.decoder_layers_config, decoder_['name'], 
                                       last_activation=last_activation) for i in self.head_names]
    
    self.input = self.encoder.input
    if self.head_count == 1:
      self.decoder = self.decoder[0]
      self.output = self.decoder(self.encoder.output)
    else:
      self.output = [ x(self.encoder.output) for x in self.decoder]
    self.model = tkm.Model(self.input, self.output)
    
    if variational:
      self.model.add_loss(self.encoder.get_variational_loss(self.input, self.output))
      self.compile(optimizer='adam', loss=None, metrics=['accuracy'])
    else:
      self.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
  @staticmethod
  def get_modules_config(network_type):
    if network_type == "ae":
      return {"class": ae_encoder, "name": "ae_encoder"}, {"class": ae_decoder, "name": "ae_decoder"}
    elif network_type == "vae":
      return {"class": vae_encoder, "name": "vae_encoder"}, {"class": ae_decoder, "name": "vae_decoder"}

# COMMAND ----------

# DBTITLE 1,Test Functions
def show_results(mnist_data, model_obj, many=False):
  n_test_samples = len(mnist_data["images"]["test"])
  index = np.random.randint(n_test_samples)
  original_image = mnist_data["images"]["test"][[index]]
  
  if many:
    reconstructed_image = model_obj.predict([mnist_data["images"]["test"][[index]] for i in range(3)])[0][0]
  else:
    reconstructed_image = model_obj.predict(mnist_data["images"]["test"][[index]])[0]

  original_image = original_image.reshape(28,28)
  reconstructed_image = reconstructed_image.reshape(28,28)
  
  #plot_reconstruction(original_image, reconstructed_image)
  
  
def test_ae(model_name, architecture, latent_dim = 2, 
            hvd_flag=False, epochs=32, heads=['position'], variational=False): 
  
  mnist_data = get_data_mnist(flat=(type(architecture.stimulus_size) is int))
  
  #ae = AE(architecture, heads, latent_dim, hvd_flag=hvd_flag, variational=variational)
  
  #jf add
  experiment_name = None
  ae = AE(architecture, experiment_name, heads, latent_dim, hvd_flag=hvd_flag, variational=variational)
  
  if hvd_flag:
    ae.fit(epochs=epochs)
  else:
    
    if ae.head_count == 1:
      x = mnist_data["images"]["train"]
      y = mnist_data["images"]["train"]
      x_ = mnist_data["images"]["train"]
      y_ = mnist_data["images"]["train"]
    else:
      x  = [mnist_data["images"]["train"] for i in heads]
      y  = [mnist_data["images"]["train"] for i in heads]
      x_ = [mnist_data["images"]["train"] for i in heads]
      y_ = [mnist_data["images"]["train"] for i in heads]
    
    ae.fit(x=x, 
           y=y, 
           epochs=epochs, 
           validation_data=(x_, y_),
           callbacks = [ae.cp_callback])
  
    ae.save_model()
  
  show_results(mnist_data, ae, many=(ae.head_count>1))
  
  return ae


def batch_tests(test_epochs, test_heads, test_hvd):
  
  # Test for: Single Node Training, VGG Autoencoder
  model_name_ae_vgg_single = NnUtils._postfix("TEST_MR_V5")
  ae_vgg_single = test_ae(model_name_ae_vgg_single, vgg_basic_architecture, 
                          epochs=test_epochs, hvd_flag=test_hvd, heads=test_heads, variational=False)

  # Test for: Single Node Training, VGG Autoencoder with a variational latent space
  model_name_vae_vgg_single = NnUtils._postfix("TEST_MR_V5")
  vae_vgg_var_single = test_ae(model_name_vae_vgg_single, vgg_basic_architecture, 
                               epochs=test_epochs, hvd_flag=test_hvd, heads=test_heads, variational=True)

  # Test for: Single Node Training, MLP Autoencoder
  model_name_ae_mlp_single = NnUtils._postfix("TEST_MR_V5")
  ae_mlp_single = test_ae(model_name_ae_mlp_single, mlp_basic_architecture, 
                          epochs=test_epochs, hvd_flag=test_hvd, heads=test_heads, variational=False)

  # Test for: Single Node Training, MLP Autoencoder with a variational latent space
  model_name_vae_mlp_single = NnUtils._postfix("TEST_MR_V5")
  ae_mlp_var_single = test_ae(model_name_vae_mlp_single, mlp_basic_architecture, 
                              epochs=test_epochs, hvd_flag=test_hvd, heads=test_heads, variational=True)

# COMMAND ----------

# DBTITLE 1,Tests for single node training - Single Head
if RUN_TESTS:
  test_epochs = 1
  test_heads = ['position']
  test_hvd = False

  batch_tests(test_epochs, test_heads, test_hvd)

# COMMAND ----------

# DBTITLE 1,Tests for single node training - Many Heads
if RUN_TESTS:
  test_epochs = 1
  test_heads = ['position','speed']
  test_hvd = False

  batch_tests(test_epochs, test_heads, test_hvd)

# COMMAND ----------

# DBTITLE 1,Tests for distributed training - Single Head
if RUN_TESTS:
  test_epochs = 1
  test_heads = ['position']
  test_hvd = True

  batch_tests(test_epochs, test_heads, test_hvd)

# COMMAND ----------

# DBTITLE 1,Tests for distributed training - Many Heads
if RUN_TESTS:
  test_epochs = 1
  test_heads = ['position','speed']
  test_hvd = True

  batch_tests(test_epochs, test_heads, test_hvd)

# COMMAND ----------

#ae_mlp.chk_dir
#dbutils.fs.ls(f'/FileStore/gb-cache/models/{model_name_single_node}/')
