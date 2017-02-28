import os
import numpy as np
import sys
from math import ceil


from keras.models import Model
from keras.layers import Dense, Dropout, Input, Masking, Activation, Embedding
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.backend as K


from RL_layer import RL_layer
from Encoder import Encoder
from Predictor import Predictor
import utils




CASES_FILENAME = "cases.txt"
QUOTES = ["'", '“', '"']




def get_data(settings):
    with open(utils.SPLITTED_SENTENCES_FILENAME, "rt") as f:
        sentences = f.read().split(utils.EOS_WORD)
    with open(utils.LABELS_FILENAME, "rt") as f:
        labels = f.read().splitlines()

    sentences = sentences[:-1]

    labels_set = set()
    result = []
    print("Reading data:\n")
    for sentence_pos in range(len(sentences)):
        if sentence_pos%1000==0:
            sys.stdout.write("\r "+str(sentence_pos)+" / "+str(len(sentences)))
        sentence = utils.strip_trailing_quotes(sentences[sentence_pos])
        sentence = sentence.strip("\n")
        sentence = sentence.strip(" ")
        parsed_sentence = sentence.split(" ")
        result.append({'label': labels[sentence_pos], "sentence": parsed_sentence})
        labels_set.add(labels[sentence_pos])
    labels_list = list(labels_set)
    labels_list.sort()
    sys.stdout.write("\n")

    from collections import Counter
    cnt = Counter([len(l['sentence']) for l in result])

    word_corpus_encode, word_corpus_decode = utils.load_word_corpus(settings['max_features'])
    settings['num_of_classes'] = len(labels_list)
    data = {'labels': labels_list,
            'sentences': result,
            'word_corpus_encode': word_corpus_encode,
            'word_corpus_decode': word_corpus_decode}
    return data, settings



def init_settings():
    settings = {}
    settings['word_embedding_size'] = 64
    settings['sentence_embedding_size'] = 128
    settings['depth'] = 6
    settings['RL_dim'] = 128
    settings['dropout_W'] = 0.2
    settings['dropout_U'] = 0.2
    settings['hidden_dims'] = [64]
    settings['dense_dropout'] = 0.5
    settings['bucket_size_step'] = 4
    settings['batch_size'] = 8
    settings['max_sentence_len_for_model'] = 16
    settings['max_sentence_len_for_generator'] = 16
    settings['max_features']=15000
    settings['with_sentences']=False
    return settings



def build_encoder(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_sentence_len_for_model'],))
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          mask_zero=True)(data_input)
    encoder = Encoder(input_dim=settings['word_embedding_size'],
                      hidden_dim=settings['sentence_embedding_size'],
                      RL_dim=settings['RL_dim'],
                      max_len=settings['max_sentence_len_for_model'],
                      batch_size=settings['batch_size'],
                      name='encoder')(embedding)
    layer = encoder

    for hidden_dim in settings['hidden_dims']:
        layer = Dense(hidden_dim)(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=data_input, output=output)
    optimizer = Adam(lr=0.001, clipnorm=5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model

def build_predictor(data, settings):
    sys.stdout.write('Building model\n')
    data_input = Input(shape=(settings['max_sentence_len_for_model'],))
    embedding = Embedding(input_dim=settings['max_features']+3,
                          output_dim=settings['word_embedding_size'],
                          mask_zero=True)(data_input)
    encoder = Predictor(input_dim=settings['word_embedding_size'],
                        hidden_dim=settings['sentence_embedding_size'],
                        RL_dim=settings['RL_dim'],
                        max_len=settings['max_sentence_len_for_model'],
                        batch_size=settings['batch_size'],
                        name='encoder')(embedding)
    layer = encoder[0]

    for hidden_dim in settings['hidden_dims']:
        layer = Dense(hidden_dim)(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(settings['dense_dropout'])(layer)
    output = Dense(settings['num_of_classes'], activation='softmax', name='output')(layer)
    model = Model(input=data_input, output=[output, encoder[1], encoder[2], encoder[3], encoder[4], encoder[5]])
    optimizer = Adam(lr=0.001, clipnorm=5)
    #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model



def build_RL_model(settings):
    x_input = Input(shape=(settings['word_embedding_size']+settings['sentence_embedding_size'],))
    h_tm1_input = Input(shape=(settings['word_embedding_size']+settings['sentence_embedding_size'],))
    action_input = Input(shape=(1,))
    layer = RL_layer(hidden_dim=settings['sentence_embedding_size'],
                     RL_dim=settings['RL_dim'])([x_input, h_tm1_input, action_input])
    model = Model(input=[x_input, h_tm1_input, action_input], output=layer)
    model.compile(loss='mse', optimizer='adam')
    return model



def build_generator_HRNN(data, settings, indexes):
    def generator():
        walk_order = list(indexes)
        np.random.shuffle(walk_order)
        batch = []
        while True:
            idx = walk_order.pop()-1
            row = data['sentences'][idx]
            sentence = row['sentence']
            label = row['label']
            if len(walk_order) == 0:
                walk_order = list(indexes)
                np.random.shuffle(walk_order)
            if len(sentence) > settings['max_sentence_len_for_generator']:
                continue

            batch.append((sentence, label))
            if len(batch)==settings['batch_size']:
                X, Y = build_batch(data, settings, batch)
                batch_sentences = batch
                batch = []

                if settings['with_sentences']:
                    yield X, Y, batch_sentences
                else:
                    yield X, Y
    return generator()

def build_batch(data, settings, sentence_batch):
    X = np.zeros((settings['batch_size'], settings['max_sentence_len_for_model']))
    Y = np.zeros((settings['batch_size'], settings['num_of_classes']), dtype=np.bool)
    for i, sentence_tuple in enumerate(sentence_batch):
        for idx, word in enumerate(sentence_tuple[0]):
            if word in data['word_corpus_encode']:
                X[i][idx] = data['word_corpus_encode'][word]+1
            else:
                X[i][idx] = settings['max_features']+1
        X[i][min(len(sentence_tuple[0]), settings['max_sentence_len_for_model']-1)] = settings['max_features']+2
        Y[i][data['labels'].index(sentence_tuple[1])] = True
    return X, Y

def run_training(data, objects):
    objects['model'].fit_generator(generator=objects['data_gen'], validation_data=objects['val_gen'], nb_val_samples=len(objects['val_indexes'])/10, samples_per_epoch=len(objects['train_indexes'])/10, nb_epoch=50, callbacks=[LearningRateScheduler(lr_scheduler)])

def lr_scheduler(epoch):
    z = epoch/50
    return z*0.0001 + (1-z)*0.001




###############################################################


def prepare_objects_RL(data, settings):
    with open(utils.INDEXES_FILENAME, "rt") as f:
        indexes = f.read().splitlines()
    indexes = [int(index) for index in indexes]
    train_segment = int(len(indexes)*0.9)

    train_indexes = indexes[:train_segment]
    val_indexes = indexes[train_segment:]

    encoder = build_encoder(data, settings)
    predictor = build_predictor(data, settings)
    rl_model = build_RL_model(settings)
    data_gen = build_generator_HRNN(data, settings, train_indexes)
    val_gen = build_generator_HRNN(data, settings, val_indexes)
    return {'encoder': encoder,
            'predictor': predictor,
            'rl_model': rl_model,
            'data_gen': data_gen,
            'val_gen': val_gen,
            'train_indexes': train_indexes,
            'val_indexes': val_indexes}





def run_training_RL(data, objects, settings):
    encoder = objects['encoder']
    predictor = objects['predictor']
    rl_model = objects['rl_model']
    epoch_size = int(len(objects['train_indexes'])/(10*settings['batch_size']))

    for epoch in range(50):
        sys.stdout.write("\nEpoch {}\n".format(epoch))
        loss1_total = []
        acc_total = []
        loss2_total = []
        for i in range(epoch_size):
            batch = next(objects['data_gen'])
            loss1 = encoder.train_on_batch(batch[0], batch[1])

            predictor.layers[2].W_emb.set_value(K.get_value(encoder.layers[2].W_emb))
            predictor.layers[2].b_emb.set_value(K.get_value(encoder.layers[2].b_emb))
            predictor.layers[2].W_R.set_value(K.get_value(encoder.layers[2].W_R))
            predictor.layers[2].b_R.set_value(K.get_value(encoder.layers[2].b_R))

            y_pred = predictor.predict(batch[0])

            output = y_pred[0]
            prev_value = y_pred[1]
            prev_prev_value = y_pred[2]
            current_value = y_pred[3]
            policy = y_pred[4]
            policy_calculated = y_pred[5]

            error = np.log(np.sum(output*batch[1], axis=1))
            X,Y = restore_exp(settings, error, prev_value, prev_prev_value, current_value, policy, policy_calculated)
            loss2 = rl_model.train_on_batch(X,Y)



            encoder.layers[2].W_S1.set_value(K.get_value(rl_model.layers[2].W_S1))
            encoder.layers[2].b_S1.set_value(K.get_value(rl_model.layers[2].b_S1))
            encoder.layers[2].W_S2.set_value(K.get_value(rl_model.layers[2].W_S2))
            encoder.layers[2].b_S2.set_value(K.get_value(rl_model.layers[2].b_S2))

            predictor.layers[2].W_S1.set_value(K.get_value(rl_model.layers[2].W_S1))
            predictor.layers[2].b_S1.set_value(K.get_value(rl_model.layers[2].b_S1))
            predictor.layers[2].W_S2.set_value(K.get_value(rl_model.layers[2].W_S2))
            predictor.layers[2].b_S2.set_value(K.get_value(rl_model.layers[2].b_S2))


            loss1_total.append(loss1[0])
            loss2_total.append(loss2)
            acc_total.append(loss1[1])
            if len(loss1_total) > 20:
                loss1_total.pop(0)
            if len(loss2_total) > 20:
                loss2_total.pop(0)
            if len(acc_total) > 20:
                acc_total.pop(0)


            sys.stdout.write("\r batch {} / {}: loss1 = {:.2f}, acc = {:.2f}, loss2 = {:.6f}".format(i, epoch_size, np.sum(loss1_total)/20, np.sum(acc_total)/20, np.sum(loss2_total)/20))




def restore_exp(settings, total_error, prev_value, prev_prev_value, current_value, policy, policy_calculated):
    DELTA = 0.9
    prev_value_input = []
    prev_prev_value_input = []
    current_value_input = []
    policy_output = []


    decision_performed = np.argwhere(policy_calculated == 1)
    for idx in decision_performed:
        pass






    #X = [np.asarray(left_input), np.asarray(bottom_input), np.asarray(action_input)]
    #Y = np.asarray(error_output)
    pass







def train(weights_filename):
    settings = init_settings()
    settings['with_sentences']=True
    data, settings = get_data(settings)
    objects = prepare_objects_RL(data, settings)
    #objects['model'].load_weights("rl.h5")
    sys.stdout.write('Compiling model\n')
    #run_training(data, objects)
    run_training_RL(data, objects, settings)
    objects['model'].save_weights(weights_filename)


if __name__=="__main__":
    train("weights.h5")





