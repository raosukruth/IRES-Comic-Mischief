import torch


processed_data_dir = "../../processed_data/"


label_to_idx = {0: 0, 1: 1}
idx_to_label = { 0: 0, 1: 1}

#label_to_idx = {'R': 0, 'PG-13': 1,  'PG':2}
#idx_to_label = { 0: 'R', 1: 'PG-13', 2: 'PG'}

genre_to_idx = {'Sci-Fi':0, 'Crime': 1, 'Romance': 2, 'Animation': 3, 'Music': 4, 'Adult': 5, 'Comedy': 6, 'War': 7, 'Horror': 8, 'Film-Noir': 9, 'Adventure': 10, 'News': 11, 'Thriller': 12, 'Western': 13, 'Mystery': 14, 'Short': 15, 'Drama': 16, 'Action': 17, 'Documentary': 18, 'History': 19, 'Family': 20, 'Fantasy': 21, 'Sport': 22, 'Biography': 23, 'Musical':4, 'Talk-Show':24}
idx_to_genre = {0:'Sci-Fi', 1:'Crime', 2:'Romance', 3:'Animation', 4:'Music,Musical', 5:'Adult', 6:'Comedy', 7:'War', 8:'Horror', 9:'Film-Noir', 10:'Adventure', 11:'News', 12:'Thriller', 13:'Western', 14:'Mystery', 15:'Short', 16:'Drama', 17:'Action', 18:'Documentary', 19:'History', 20:'Family', 21:'Fantasy', 22:'Sport', 23:'Biography', 24:'Musical',25:'Talk-Show'}

idx_to_emotion = {0: 'positive', 1: 'sadness', 2: 'joy', 3: 'trust', 4: 'fear', 5: 'negative', 6: 'surprise', 7: 'anger', 8: 'anticipation', 9: 'disgust'}
emotion_to_idx = {'positive': 0, 'sadness': 1, 'joy': 2, 'trust': 3, 'fear': 4, 'negative': 5, 'surprise': 6, 'anger': 7, 'anticipation': 8, 'disgust': 9}

fold = "1"


training_features = processed_data_dir + 'train_features_lrec_camera.json'
val_features = processed_data_dir + 'val_features_lrec_camera.json'
test_features = processed_data_dir + 'test_features_lrec_camera.json' 


batch_size = 16

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

