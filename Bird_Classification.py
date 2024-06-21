import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import librosa
import librosa.display

# Load the image classification model
model = load_model('./image_model.h5',compile=False)
lab = {0:'ABBOTTS BABBLER', 1: 'ABBOTTS BOBBY', 2: 'ABYSSINIAN GROUND HORNBILL', 3: 'AFRICAN CROWNED CRANE', 4: 'ALBATROSS', 5: 'ALBERTS TOWHEE', 6: 'ALEXANDRINE PARAKEET', 7: 'ALEXANDRINE PARAKEET', 8: 'ALPINE CHOUGH', 9: 'ALTAMIRA YELLOWTHROAT', 10: 'AMERICAN AVOCET', 11: 'AMETHYST WOODSTAR', 12: 'APAPANE', 13: 'APOSTLEBIRD', 14: 'ARARIPE MANAKIN', 15: 'ASHY STORM PETREL', 16: 'ASHY THRUSHBIRD', 17:  'ASIAN CRESTED IBIS', 18: 'ASIAN DOLLARD BIRD', 19: 'ASIAN GREEN BEE EATER', 20: 'ASIAN OPENBILL STORK', 21: 'AUCKLAND SHAQ', 22: 'AUSTRAL CANASTERO', 23: 'AUSTRALASIAN FIGBIRD', 24: 'AVADAVAT', 25: 'AZARAS SPINETAIL', 26: 'AZURE BREASTED PITTA', 27: 'AZURE JAY', 28: 'AZURE TANAGER', 29: 'AZURE TIT', 30: 'BEARDED BARBET', 31: 'BEARDED BELLBIRD', 32: 'BEARDED REEDLING', 33: 'BELTED KINGFISHER', 34: 'BIRD OF PARADISE', 35: 'BLACK SWAN', 36: 'BLONDE CRESTED WOODPECKER', 37: 'BLOOD PHEASANT', 38: 'BLUE COAU', 39: 'BLUE DACNIS', 40: 'BLUE GRAY GNATCATCHER', 41: 'BLUE GROSBEAK', 42: 'BLUE GROUSE', 43: 'BLUE HERON', 44: 'BLUE MALKOHA', 45: 'BLUE THROATED PIPING GUAN', 46: 'BLUE THROATED TOUCANET', 47: 'BOBOLINK', 48: 'BORNEAN BRISTLEHEAD', 49: 'BORNEAN LEAFBIRD', 50: 'BORNEAN PHEASANT', 51: 'BRANDT CORMARANT', 52: 'BREWERS BLACKBIRD', 53: 'BROWN CREPPER', 54: 'BROWN HEADED COWBIRD', 55: 'BROWN NOODY', 56: 'BROWN THRASHER', 57: 'BUFFLEHEAD', 58: 'BULWERS PHEASANT', 59: 'BURCHELLS COURSER', 60: 'BUSH TURKEY', 61: 'CEDAR WAXWING', 62: 'CERULEAN WARBLER', 63: 'CHARA DE COLLAR', 64: 'CHATTERING LORY', 65: 'CHESTNET BELLIED EUPHONIA', 66: 'CHESTNUT WINGED CUCKOO', 67: 'CHINESE BAMBOO PARTRIDGE', 67: 'CHINESE POND HERON', 68:'CHIPPING SPARROW', 69: 'CHUCAO TAPACULO', 70: 'CHUKAR PARTRIDGE', 71: 'CINNAMON ATTILA', 72: 'CINNAMON FLYCATCHER', 73: 'CINNAMON TEAL', 74: 'CLARKS GREBE', 75: 'CLARKS NUTCRACKER', 76: 'DALMATIAN PELICAN', 77: 'DARJEELING WOODPECKER', 78: 'DARK EYED JUNCO', 79: 'D-ARNAUDS BARBET', 80: 'DAURIAN REDSTART', 81: 'DEMOISELLE CRANE', 82: 'DOUBLE BARRED FINCH', 83: 'DOUBLE BRESTED CORMARANT', 84: 'DOUBLE EYED FIG PARROT', 85: 'DOWNY WOODPECKER', 86: 'DUNLIN', 87: 'DUSKY LORY', 88: 'DUSKY ROBIN', 89: 'ECUADORIAN HILLSTAR', 90: 'EGYPTIAN GOOSE', 91: 'ELEGANT TROGON', 92: 'ELLIOTS  PHEASANT', 93: 'EMERALD TANAGER', 94: 'EMPEROR PENGUIN', 95: 'EMU', 96: 'ENGGANO MYNA', 97: 'EURASIAN BULLFINCH', 98: 'EURASIAN GOLDEN ORIOLE', 99: 'EURASIAN MAGPIE', 100: 'EUROPEAN GOLDFINCH', 101: 'EUROPEAN TURTLE DOVE', 102: 'EVENING GROSBEAK', 103: 'FAIRY BLUEBIRD', 104: 'FAIRY PENGUIN', 105: 'FAIRY TERN', 106: 'FAN TAILED WIDOW', 107: 'FASCIATED WREN', 108: 'FIERY MINIVET', 109: 'FIORDLAND PENGUIN', 110: 'FIRE TAILLED MYZORNIS', 111: 'FLAME BOWERBIRD', 112: 'FLAME TANAGER', 113: 'FOREST WAGTAIL', 114: 'FRIGATE', 115: 'FRILL BACK PIGEON', 116: 'HAMERKOP', 117: 'HARLEQUIN DUCK', 118: 'HARLEQUIN QUAIL', 119: 'HARPY EAGLE', 120: 'HAWAIIAN GOOSE', 121: 'HAWFINCH', 122: 'HELMET VANGA', 123: 'HEPATIC TANAGER', 124: 'HIMALAYAN BLUETAIL', 125: 'HIMALAYAN MONAL', 126: 'HOATZIN',  127: 'HOODED MERGANSER', 128: 'HOOPOES', 129: 'HORNED GUAN', 130: 'HORNED LARK', 131: 'HORNED SUNGEM', 132: 'HOUSE FINCH', 133: 'HOUSE SPARROW', 134: 'HYACINTH MACAW', 135: 'IBERIAN MAGPIE', 136: 'IBISBILL', 137: 'IMPERIAL SHAQ', 138: 'INCA TERN', 139: 'INDIAN BUSTARD', 140: 'INDIAN PITTA', 141: 'INDIAN ROLLER', 142: 'INDIAN VULTURE', 143: 'INDIGO BUNTING', 144: 'INDIGO FLYCATCHER', 145: 'INLAND DOTTEREL', 146: 'IVORY BILLED ARACARI', 147: 'IVORY GULL', 148: 'IWI', 149: 'JABIRU', 150: 'JACK SNIPE', 151: 'JACOBIN PIGEON', 152: 'JANDAYA PARAKEET', 153: 'JAPANESE ROBIN', 154: 'JAVA SPARROW', 155: 'JOCOTOCO ANTPITTA', 156: 'KAGU', 157: 'LARK BUNTING', 158: 'LAUGHING GULL', 159: 'LAZULI BUNTING', 160: 'LESSER ADJUTANT', 161: 'LILAC ROLLER', 162: 'LIMPKIN', 163: 'LITTLE AUK', 164: 'LOGGERHEAD SHRIKE', 165: 'LONG-EARED OWL', 166: 'LOONEY BIRDS', 167: 'LUCIFER HUMMINGBIRD', 168: 'MAGPIE GOOSE', 169: 'MALABAR HORNBILL', 170: 'MALACHITE KINGFISHER', 171: 'MALAGASY WHITE EYE', 172: 'MALEO', 173: 'MALLARD DUCK', 174: 'MANDRIN DUCK', 175: 'MANGROVE CUCKOO', 176: 'MARABOU STORK', 177: 'MASKED BOBWHITE', 178: 'MASKED BOOBY', 179: 'MASKED LAPWING', 180: 'MCKAYS BUNTING', 181: 'MERLIN', 182:  'MIKADO  PHEASANT', 183: 'MILITARY MACAW', 184: 'MOURNING DOVE', 185: 'MYNA', 186: 'NICOBAR PIGEON', 187: 'NOISY FRIARBIRD', 188: 'NORTHERN BEARDLESS TYRANNULET', 189: 'NORTHERN CARDINAL', 190: 'NORTHERN FLICKER', 191: 'NORTHERN FULMAR', 192: 'NORTHERN GANNET', 193: 'NORTHERN GOSHAWK', 194: 'NORTHERN JACANA', 195: 'NORTHERN MOCKINGBIRD', 196: 'NORTHERN PARULA', 197: 'NORTHERN RED BISHOP', 198: 'NORTHERN SHOVELER', 199: 'OCELLATED TURKEY'}

# Load the audio classification model
audio_model = load_model('./audio_model.h5', compile=False)
audio_labels ={0: 'Andean Guan_sound', 1: 'Andean Tinamou_sound', 2: 'Australian Brushturkey_sound', 3: 'Band-tailed Guan_sound', 4: 'Barred Tinamou_sound', 5: 'Bartletts Tinamou_sound', 6: 'Baudo Guan_sound', 7: 'Bearded Guan_sound', 8: 'Berlepschs Tinamou_sound', 9: 'Biak Scrubfowl_sound', 10: 'Black Tinamou_sound', 11: 'Black-billed Brushturkey_sound', 12: 'Black-capped Tinamou_sound', 13: 'Black-fronted Piping Guan_sound', 14: 'Blue-throated Piping Guan_sound', 15: 'Brazilian Tinamou_sound', 16: 'Brown Tinamou_sound', 17: 'Brushland Tinamou_sound', 18: 'Buff-browed Chachalaca_sound', 19: 'Cauca Guan_sound', 20: 'Chaco Chachalaca_sound', 21: 'Chestnut-bellied Guan_sound', 22: 'Chestnut-headed Chachalaca_sound', 23: 'Chestnut-winged Chachalaca_sound', 24: 'Chilean Tinamou_sound', 25: 'Choco Tinamou_sound', 26: 'Cinereous Tinamou_sound', 27: 'Collared Brushturkey_sound', 28: 'Colombian Chachalaca_sound', 29: 'Common Ostrich_sound', 30: 'Crested Guan_sound', 31: 'Curve-billed Tinamou_sound', 32: 'Darwins Nothura_sound', 33: 'Dusky Megapode_sound', 34: 'Dusky-legged Guan_sound', 35: 'Dwarf Cassowary_sound', 36: 'Dwarf Tinamou_sound', 37: 'East Brazilian Chachalaca_sound', 38: 'Elegant Crested Tinamou_sound', 39: 'Emu_sound', 40: 'Great Spotted Kiwi_sound', 41: 'Great Tinamou_sound', 42: 'Greater Rhea_sound', 43: 'Grey Tinamou_sound', 44: 'Grey-headed Chachalaca_sound', 45: 'Grey-legged Tinamou_sound', 46: 'Highland Tinamou_sound', 47: 'Hooded Tinamou_sound', 48: 'Huayco Tinamou_sound', 49: 'Lesser Nothura_sound', 50: 'Lesser Rhea_sound', 51: 'Little Chachalaca_sound', 52: 'Little Spotted Kiwi_sound', 53: 'Little Tinamou_sound', 54: 'Maleo_sound', 55: 'Malleefowl_sound', 56: 'Marail Guan_sound', 57: 'Melanesian Megapode_sound', 58: 'Micronesian Megapode_sound', 59: 'Moluccan Megapode_sound', 60: 'New Guinea Scrubfowl_sound', 61: 'Nicobar Megapode_sound', 62: 'North Island Brown Kiwi_sound', 63: 'Northern Cassowary_sound', 64: 'Okarito Kiwi_sound', 65: 'Orange-footed Scrubfowl_sound', 66: 'Ornate Tinamou_sound', 67: 'Pale-browed Tinamou_sound', 68: 'Patagonian Tinamou_sound', 69: 'Philippine Megapode_sound', 70: 'Plain Chachalaca_sound', 71: 'Puna Tinamou_sound', 72: 'Quebracho Crested Tinamou_sound', 73: 'Red-billed Brushturkey_sound', 74: 'Red-faced Guan_sound', 75: 'Red-legged Tinamou_sound', 76: 'Red-throated Piping Guan_sound', 77: 'Red-winged Tinamou_sound', 78: 'Rufous-bellied Chachalaca_sound', 79: 'Rufous-headed Chachalaca_sound', 80: 'Rufous-vented Chachalaca_sound', 81: 'Rusty Tinamou_sound', 82: 'Rusty-margined Guan_sound', 83: 'Scaled Chachalaca_sound', 84: 'Slaty-breasted Tinamou_sound', 85: 'Small-billed Tinamou_sound', 86: 'Solitary Tinamou_sound', 87: 'Somali Ostrich_sound', 88: 'Southern Brown Kiwi_sound', 89: 'Southern Cassowary_sound', 90: 'Speckled Chachalaca_sound', 91: 'Spixs Guan_sound', 92: 'Spotted Nothura_sound', 93: 'Sula Megapode_sound', 94: 'Taczanowskis Tinamou_sound', 95: 'Tanimbar Megapode_sound', 96: 'Tataupa Tinamou_sound', 97: 'Tawny-breasted Tinamou_sound', 98: 'Tepui Tinamou_sound', 99: 'Thicket Tinamou_sound', 100: 'Tongan Megapode_sound', 101: 'Trinidad Piping Guan_sound', 102: 'Undulated Tinamou_sound', 103: 'Vanuatu Megapode_sound', 104: 'Variegated Tinamou_sound', 105: 'Wattled Brushturkey_sound', 106: 'West Mexican Chachalaca_sound', 107: 'White-bellied Chachalaca_sound', 108: 'White-bellied Nothura_sound', 109: 'White-browed Guan_sound', 110: 'White-crested Guan_sound', 111: 'White-throated Tinamou_sound', 112: 'White-winged Guan_sound', 113: 'Yellow-legged Tinamou_sound'}


def processed_img(location):
    img=load_img(location,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def process_audio(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract mel spectrogram features
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    target_shape = (224, 224)  # Target shape of the spectrogram
    mel_spectrogram_resized = np.resize(mel_spectrogram_db, target_shape)
    processed_input = np.expand_dims(mel_spectrogram_resized, axis=-1)  # Add channel dimension
     
    # Repeat the input to have 3 channels (assuming the model expects 3-channel input)
    processed_input = np.repeat(processed_input, 3, axis=-1)
    processed_input = np.expand_dims(processed_input, axis=0)  # Add batch dimension
    
    # Predict the class
  
    prediction = audio_model.predict(processed_input)
    predicted_class = np.argmax(prediction)
    predicted_label = audio_labels[predicted_class]
    
    return predicted_label

def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Birds Species Classification")
    # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',
    #             unsafe_allow_html=True)
    
    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Bird is: "+result)
    # File uploader for audio
    audio_file = st.file_uploader("Choose an Audio of Bird", type=["mp3", "wav"])
    if audio_file is not None:
        # Button to predict audio
        if st.button("Predict Audio"):
            with open('./upload_audio/' + audio_file.name, "wb") as f:
                f.write(audio_file.getbuffer())
            result = process_audio('./upload_audio/' + audio_file.name)
            st.success("Predicted Bird from Audio: " + result)
    
run()