# Databricks notebook source
# MAGIC %run ./base_autoencoder

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, average_precision_score

# COMMAND ----------

class History:
  def __init__(self):
    
    self.losses = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    self.pr_curves = []
  
    self.last_precision = 0.01
    
  def capture_history(self, new_history):
    
    for k,v in new_history.items():
      if k in self.losses:
        self.losses[k] += v
    
  def capture_pr(self, model, test_images, test_labels, index):
    y_true = test_labels.numpy().ravel()
    y_scores = model.predict(test_images)[:,index].ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    average_precision = average_precision_score(y_true, y_scores)
    
    self.pr_curves.append({'index': index,
                           'precision': precision,
                           'recall': recall,
                           'thresholds': thresholds,
                           'average_precision': average_precision,
                           'y_scores': y_scores})
    
    self.last_precision = average_precision
    
  def plot_history(self):
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
      x = np.linspace(0.01, 1)
      y = (f_score * x) / (2 * x - f_score)
      l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
      plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    r_scale = np.linspace(0.0,1.0,len(self.pr_curves))

    for k, pr_curve in enumerate(self.pr_curves):
      plt.plot(pr_curve['recall'], pr_curve['precision'], c=[r_scale[k],0.0,0.0])

    plt.ylabel('Precision')
    plt.xlabel('Recall')

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.subplot(1,2,2)
    average_precision = [x['average_precision'] for x in self.pr_curves]
    plt.scatter(range(len(average_precision)), average_precision, c=[[r, 0.0, 0.0] for r in r_scale], s=80)
    plt.ylabel('Average Precision')
    plt.xlabel('Epoch Number')

# COMMAND ----------

class TL(History, ClassifierBase):
  def __init__(self, architecture, experiment_name, latent_dim, heads, train_labels, dropout_rate=0.1):
    
    super().__init__()
    
    self.autoencoder = AE(architecture, experiment_name, heads, latent_dim,
                          hvd_flag=False, variational=True)
    
    self.latent_dim = latent_dim
    self.dropout_rate = dropout_rate
    
    self._encoder_network = self.autoencoder.encoder
    
    self.trainable_layer_names = {x.name:k+1 for k,x in enumerate(self._encoder_network.model.layers) if len(x.trainable_weights)>0}
    
    self.create_classifier(train_labels)
    
  def create_classifier(self, train_labels):
    
    output_bias, self.class_weight = self.get_bias(train_labels)
    
    x = tkl.Concatenate()([self._encoder_network.z_mean, self._encoder_network.z_log_sigma])
    
    x = tkl.Dense(units=self.latent_dim)(x)
    x = tkl.BatchNormalization()(x)
    x = tkl.ReLU()(x)
    x = tkl.Dropout(self.dropout_rate)(x)
    
    classifier_output = tkl.Dense(1, activation="sigmoid", 
                                     bias_initializer=output_bias)(x)

    self.classifier_model = tkm.Model(self._encoder_network.input, classifier_output)

    self.classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy'])
    
  def get_freeze_bias(self, layer_name):
    deno = len(self.trainable_layer_names) / 4
    
    return 1 - (np.exp(-self.trainable_layer_names[layer_name]*self.last_precision/deno))
  
  def freeze(self):
    #self._encoder_network.trainable = False
    for layer in transfer_model._encoder_network.model.layers:
      if random.random() > 0.20:
        layer.trainable = False
    
  def unfreeze(self):
    #self._encoder_network.trainable = True
    for layer in transfer_model._encoder_network.model.layers:
      if random.random() > 0.2:#self.train_bias[layer.name]:#> 0.20:
        layer.trainable = True
    
  def train(self, train_images, train_labels, test_images, test_labels, 
                  epoch_block_count=20, 
                  ae_epoch_block_size=5, ae_batch_size=128,
                  cl_epoch_block_size=50, cl_batch_size=8192):
    
    for _block in range(epoch_block_count):
      self.unfreeze()
      
      self.autoencoder.fit(train_images, train_images, 
                           validation_data=(test_images, test_images), 
                           epochs=ae_epoch_block_size,
                           batch_size=ae_batch_size,
                           verbose=False)
      
      self.ae_history = self.capture_history(self.ae_history, self.autoencoder.model.history.history)
      
      self.freeze()
      
      self.classifier_model.fit(train_images, train_labels, 
                                validation_data=(test_images, test_labels), 
                                epochs=cl_epoch_block_size,
                                batch_size=cl_batch_size, # MAKE THIS REALLY BIG LIKE ... 50% OF THE TOTAL SAMPLE SIZE
                                class_weight=self.class_weight,
                                verbose=False)
      
      self.cl_history = self.capture_history(self.cl_history, self.classifier_model.history.history)
      
      self.capture_pr(self.classifier_model, test_images, test_labels)
      
      print(f"Completed block {_block} of {epoch_block_count}")

# COMMAND ----------

vgg_basic_architecture.encoder
