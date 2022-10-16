"""DSSM based on tensorflow """
#===========================
#测试代码
#===========================
import datetime
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import numpy as np

# load data
def get_dataset(self, file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=2,  # 为了示例更容易展示，手动设置较小的值
        # label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset

train_epoch = 3
#data_path = "./dataset/positive.csv"
# get data from csv
# sel.data = get_dataset(data_path)
# get data from tensorflow dataset
ratings = tfds.load("movie_lens/100k-ratings", split="train")
data = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
    "user_gender": int(x["user_gender"]),
    "user_zip_code": x["user_zip_code"],
    "user_occupation_text": x["user_occupation_text"],
    "bucketized_user_age": int(x["bucketized_user_age"]),
})

# negative sample
# 数据的取值对应ItemModel中各项特征!!
candidates = ratings.map(lambda x: {
    'movie_id': x['movie_id'],
    "bucketized_user_age": x["bucketized_user_age"]
})

tf.random.set_seed(42)

# create vocabularies
feature_names = ["movie_id", "user_id", "user_gender", "user_zip_code",
                 "user_occupation_text", "bucketized_user_age"]
vocabularies = {}
for feature_name in feature_names:
    vocab = data.batch(10000).map(lambda x: x[feature_name])
    vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))  # 构造查询表

# train data
shuffled = data.shuffle(1000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(800)
test = shuffled.skip(800).take(200)

cached_train = train.shuffle(1000).batch(128)
cached_test = test.batch(128).cache()

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_dimension = 32
        self.categorical_features_str = ["user_id", "user_zip_code"]
        self.categorical_features_int = ["user_gender", "bucketized_user_age"]
        self.continuous_features_standardization = ["bucketized_user_age"]
        # get all features' embeddings
        self.all_features = self.categorical_features_str + \
                             self.categorical_features_int + \
                             self.continuous_features_standardization
        self._embeddings = {}
        self.get_embeddings()

    def call(self, inputs):
        embeddings = []
        for name in self.all_features:
            if name in self.continuous_features_standardization:
                embeddings.append(tf.reshape(self._embeddings[name](inputs[name]), (-1, 1)))
            else:
                embedding_fn = self._embeddings[name]
                embeddings.append(embedding_fn(inputs[name]))

        return tf.concat(embeddings, axis=1)

    def get_embeddings(self):
        # str categorical features
        for feature_name in self.categorical_features_str:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.StringLookup(
                    vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        # int categorical features
        for feature_name in self.categorical_features_int:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        # continuous features
        for feature_name in self.continuous_features_standardization:
            self._embeddings[feature_name] = tf.keras.layers.Normalization(axis=None)

    def get_continuous_discretization(self):
        pass

class ItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_dimension = 32
        self.categorical_features_str = ["movie_id"]
        self.categorical_features_int = []
        self.continuous_features_standardization = ["bucketized_user_age"]
        # get all features' embeddings
        self.all_features = self.categorical_features_str + \
                             self.categorical_features_int + \
                             self.continuous_features_standardization
        self._embeddings = {}
        self.get_embeddings()

    def call(self, inputs):
        embeddings = []
        for name in self.all_features:
            if name in self.continuous_features_standardization:
                embeddings.append(tf.reshape(self._embeddings[name](inputs[name]), (-1, 1)))
            else:
                embedding_fn = self._embeddings[name]
                embeddings.append(embedding_fn(inputs[name]))

        return tf.concat(embeddings, axis=1)

    def get_embeddings(self):
        # str categorical features
        for feature_name in self.categorical_features_str:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.StringLookup(
                    vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        # int categorical features
        for feature_name in self.categorical_features_int:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        # continuous features
        for feature_name in self.continuous_features_standardization:
            self._embeddings[feature_name] = tf.keras.layers.Normalization(axis=None)

    def get_continuous_discretization(self):
        pass

    def get_continuous_feature_standardization(self, feature_name):
        normalization = tf.keras.layers.Normalization(axis=None)
        normalization.adapt(data.map(lambda x: x[feature_name]))  # data为全局数据集，以构造更完整的查询表
        # normalization.adapt(self.data.map(lambda x: x[feature_name])).batch(1024))??
        return normalization

class DSSMForRecall(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.user_model = tf.keras.Sequential([
            UserModel(),
            tf.keras.layers.Dense(32)
        ])
        self.item_model = tf.keras.Sequential([
            ItemModel(),
            tf.keras.layers.Dense(32)
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates.batch(128).map(self.item_model)  # 负采样的候选集
            )
        )

    def compute_loss(self, input, training=False) -> tf.Tensor:
        user_input = {}
        for feature_name in self.user_model.layers[0].all_features:
            user_input[feature_name] = input[feature_name]
        user_embeddings = self.user_model(user_input)

        item_input = {}
        for feature_name in self.item_model.layers[0].all_features:
            item_input[feature_name] = input[feature_name]
        item_embeddings = self.item_model(item_input)

        return self.task(user_embeddings, item_embeddings)


# init model
model = DSSMForRecall()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# tensorboard
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train
model.fit(
    cached_train,
    epochs=train_epoch,
    callbacks=[tensorboard_callback]
)

# evaluate
train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")
