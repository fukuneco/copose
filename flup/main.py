import base64
import io
import cv2
import keras
import numpy as np
from PIL import Image
from keras.backend import tensorflow_backend as backend
from django.conf import settings
import copy
from Mod.mytmpMod import *
####################################################################################
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
####################################################################################
IMG_DIR = "./001_images/"
LOAD_DIR = "./001_results/"
RANK_SAVE_DIR = "./002_ranking/"                       # ランキング結果の保存先
pick_file= "001_best_human_joints.pickle"

RANK_NUM = 3 #上位何枚まで表示するか

MODEL = []


# @upload_image: アップロードされたイメージパス？
def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''

  
    # 1 start
    # 設定からモデルファイルのパスを取得
    model_file_path = settings.MODEL_FILE_PATH
   
    # kerasでモデルを読み込む
    model = create_model()
    model.load_weights(model_file_path)
    #model = keras.models.load_model(model_file_path) # 20.12.22kerasモデルはないので省く
    # 1 end
    
    """

    # 2 start ---
    # https://stackoverflow.com/questions/51911088/how-to-get-s3-directory-as-os-path-in-python-with-boto3/51911893
    ############# heroku + AWS S3の場合
    #こっちは読み込めた
    global MODEL
    #if MODEL ==[]: #初回のみ読み込ませたい
    import boto3
    print("model downloading ...")
    s3 = boto3.resource('s3')
    s3.meta.client.download_file('mymodeltest', 'model.h5', '/tmp/model.h5')
    print("model leading ...")
    MODEL = create_model()
    MODEL.load_weights('/tmp/model.h5')
    model = copy.deepcopy(MODEL)
    model.summary()
    #else:
    #    print("model already loaded")
    #    model = copy.deepcopy(MODEL)
    #    model.summary()
    #############
    # 2 end---
    """
    # アップロードされた画像ファイルをメモリ上でOpenCVのimageに格納
    #image = np.asarray(Image.open(upload_image))
    #loaded_img = cv2.imread(upload_image) #oriImg
    loaded_img = np.asarray(Image.open(upload_image))
    loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
    your_joints = getJoints(model, loaded_img) #(画像)
    print(your_joints)

    # 関節位置を描画
    draw_img = loaded_img
    for i in range(18):#関節数ループ
        if(np.isnan(your_joints[i][0])):
            continue
        cv2.circle(draw_img, (int(your_joints[i][0]), int(your_joints[i][1])), 5, colors[i], thickness=-1)

    # 比較対象の読み込み
    loaded_joints = pickle_load(LOAD_DIR+pick_file)

    # 類似ファイル名の取得 ##################################
    df_sim = getSimFileList(loaded_joints, your_joints)
    #print(df_sim["filename"][1])

    """
    # 画像をOpenCVのBGRからRGB変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 画像をRGBからグレースケール変換
    image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # カスケードファイルの読み込み
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCVを利用して顔認識
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(64, 64))

    # 顔が１つ以上検出できた場合
    if len(face_list) > 0:
        count = 1
        for (xpos, ypos, width, height) in face_list:
            # 認識した顔の切り抜き
            face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]
            # 切り抜いた顔が小さすぎたらスキップ
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                continue
            # 認識した顔のサイズ縮小
            face_image = cv2.resize(face_image, (64, 64))
            # 認識した顔のまわりを赤枠で囲む
            cv2.rectangle(image_rgb, (xpos, ypos),
                          (xpos+width, ypos+height), (0, 0, 255), thickness=2)
            # 認識した顔を1枚の画像を含む配列に変換
            face_image = np.expand_dims(face_image, axis=0)

 
            # 認識した顔から名前を特定
            name, result = detect_who(model, face_image)

            # 認識した顔に名前を描画
            cv2.putText(image_rgb, f"{count}. {name}", (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            # 結果をリストに格納
            result_list.append(result)
            count = count + 1
    """
    # ランキング上位画像を取得########################################################
    result_img_list = []
    #最初に元画像を追加
    is_success, img_buffer = cv2.imencode(".png", draw_img)
    if is_success:
        io_buffer = io.BytesIO(img_buffer)
        d_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")
    result_img_list.append(d_img)

    result_list = []
    for i in range(RANK_NUM):
        result_list.append(df_sim["filename"][i])

        rank_img = np.asarray(Image.open(IMG_DIR+df_sim["filename"][i]))
        rank_img=cv2.cvtColor(rank_img, cv2.COLOR_BGR2RGB)

        # 画像をPNGに変換
        #is_success, img_buffer = cv2.imencode(".png", draw_img)
        is_success, img_buffer = cv2.imencode(".png", rank_img)
        if is_success:
            # 画像をインメモリのバイナリストリームに流し込む
            io_buffer = io.BytesIO(img_buffer)
            # インメモリのバイナリストリームからBASE64エンコードに変換
            #result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")
            result_img_list.append(base64.b64encode(io_buffer.getvalue()).decode().replace("'", ""))
    ###############################################################################

    # tensorflowのバックエンドセッションをクリア
    backend.clear_session()
    print("tf finished...")
    # 結果を返却
    #return (result_list, result_name, result_img)
    #return ( ["test","test2"], result_name, result_img)  #とりあえずこれで動く
    #return ( [df_sim["filename"][0], df_sim["filename"][1], df_sim["filename"][2]], result_name, result_img)
    return ( result_list, result_name, result_img_list)