import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/unnati-xyz/music-generation/blob/16438a4d4ecb4645007be2d87a053cc904ebe780/data_utils/parse_files.py#L196
def time_blocks_to_fft_blocks(blocks_time_domain):
    fft_blocks = []
    plot_block=[]
    amplitude = []
    for block in blocks_time_domain:
        # Computes the one-dimensional discrete Fourier Transform and returns the complex nD array
        # i.e The truncated or zero-padded input, transformed along the axis indicated by axis, or the last one if axis is not specified.
        fft_block = np.fft.fft(block)
        new_block = np.concatenate(
                (np.real(fft_block), np.imag(fft_block)))  # Joins a sequence of arrays along an existing axis.
        fft_blocks.append(new_block)
    
    return np.array(fft_blocks)

def to_sequences(dataset, seq_size=1, output_len=300, stride = 150):
    x = []
    y = []
    for i in range(0,len(dataset)-seq_size-output_len,stride):
        window = dataset[i:(i+seq_size)]
        x.append(window)
        y.append(dataset[i+seq_size:i+seq_size+output_len])
        
    return np.array(x),np.array(y)

def chunk(dataset, seq_size=1, stride = 1):
    x = []
    for i in range(0,len(dataset)-seq_size,stride):
        window = dataset[i:(i+seq_size)]
        x.append(window)
        
    return np.array(x)

def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] // 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return np.array(time_blocks)


def preprocess_pipeline(category = 'other', start = 0, end = 10, \
                        directory = '../data/wav_csvs/seperation', trainTest = True, \
                        chunk_size = 500, input_len = 20, output_len = 20):

    # get data
    print("beginning loading data")
    order = ['bass', 'drums', 'other', 'piano', 'vocals']
    category_of_focus = order.index(category)
    
    sep_df = [[] for _ in order]
    count=0
    for x in range(start, end):
        print(x)
        for index, cat in enumerate(order):
            if(index == category_of_focus):
                print(index)
                sep_df[index].append(pd.read_csv(directory + '/' + cat + str(x) + '.csv', index_col=0))
    

    # concatenating data
    print("concatenating data")
    MAX_VAL = 32768
    means = []
    # total = pd.concat(total_df) / MAX_VAL
    sep = []

    for index in range(len(order)):
        if(index == category_of_focus):
            sep.append(pd.concat(sep_df[index]) / MAX_VAL)
        else:
            sep.append(pd.DataFrame())
    del sep_df

    # combining into one numpy array
    print("combinging into one np dataframe")
    num_songs = len(sep[category_of_focus])
    duration = len(sep[category_of_focus].iloc[0])
    seq_combined = np.zeros((num_songs, duration, 5))
    for s in range(num_songs):
        for c in range(5):
            if (c == category_of_focus):
                seq_combined[s, :, c] = sep[c].iloc[s]
    del sep

    # creating chunks
    print("creating chunks")
    
    fftb = []
    for x in range(len(seq_combined)):
        fftb.append(time_blocks_to_fft_blocks(chunk(seq_combined[x, :, category_of_focus], chunk_size, chunk_size)))


    # sequencing
    print("sequencing")
    fft_combined = np.array(fftb)
    
    
    stride = 1

    x_sequences = []
    y_sequences = []
    for x in range(num_songs):
        temp_x, temp_y = to_sequences(fft_combined[x,:,:], seq_size = input_len, output_len = output_len, stride = stride)
        x_sequences.append(temp_x)
        y_sequences.append(temp_y)
    del fftb
    del fft_combined
    len(x_sequences[0])

    # trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.10, random_state=42)
    x_sequences = np.concatenate(x_sequences)
    y_sequences = np.concatenate(y_sequences)

    if(trainTest):
        np.random.seed(42) 
        num_data = len(x_sequences)
        x_test = x_sequences[int(0.9*num_data):]
        y_test = x_sequences[int(0.9*num_data):]
        x_train = x_sequences[:int(0.9*num_data)]
        y_train = x_sequences[:int(0.9*num_data)]

        np.random.shuffle(x_test)
        np.random.shuffle(y_test)
        np.random.shuffle(x_train)
        np.random.shuffle(y_train)

        del x_sequences
        del y_sequences

        return x_train, y_train, x_test, y_test
    else:
        return x_sequences, y_sequences

def plot_prediction(x_true, y_true, model):
    

    x_time = fft_blocks_to_time_blocks(x_true).flatten()
    y_time = fft_blocks_to_time_blocks(y_true).flatten()
    pred = model.predict(np.array([x_true]))
    pred_time = fft_blocks_to_time_blocks(pred).flatten()

    x_len = len(x_time)
    y_len = len(y_time)
    plt.plot(np.arange(0,x_len), x_time, label = "input")
    plt.plot(np.arange(x_len,x_len+y_len), y_time, label = "real")
    plt.plot(np.arange(x_len,x_len+y_len), pred_time, label = "pred")
    plt.legend()
    plt.show()
    return np.concatenate((x_time, pred_time))

def predict_many(x_true, model, num_predictions = 30, overlap=5):
    curr_input = x_true
    curr_song = fft_blocks_to_time_blocks(x_true).flatten()
    for x in range(num_predictions):
        prediction = model.predict(np.array([curr_input]))[0]
        curr_song = np.concatenate((curr_song, fft_blocks_to_time_blocks(prediction[0:overlap]).flatten()))
        curr_input = np.concatenate((curr_input[5:], prediction[0:overlap, :]))
    plt.plot(curr_song)
    return curr_song