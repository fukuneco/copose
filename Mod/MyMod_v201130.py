import math
import pickle
import os
import shutil
#フォルダ作成###############################
def mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    try:
        os.makedirs(dir_path)
        #os.makedirs(dir_path, exist_ok = True)
    except:
        import time
        print ('フォルダが使用中です. 再アクセス中...')
        time.sleep(2)
        os.makedirs(dir_path)
        print ('作成しました')
#######################################################

# パスからフォルダ内部のファイル名を取得
def getFileName(img_dir):
    import os
    loaded_filename = []
    for file in os.listdir(img_dir):
        base, ext = os.path.splitext(file) # ファイル名と拡張子を分離
        if ext == '.jpg' or ext == '.png':
            loaded_filename.append(file)
            
    #print(loaded_filename)

    return loaded_filename

# 人のidxから関節座標を取得
def getHumanJoint(best_human_idx, subset, candidate, path):
    best_human_pos = {}
    for i in range(18):
        index = subset[best_human_idx][i] 
        if -1 in [index]:                           # 関節が見つからない場合-1が代入されている
            continue

        Y = candidate[index.astype(int), 1]
        X = candidate[index.astype(int), 0]

        #best_human_pos[i] = (LOAD_IMG_PATH+',%d' %i, X , Y)              # 関節idx, x座標， y座標を保存
        #p = (path+',%d' %i)
        best_human_pos[i] = (X, Y)
    return best_human_pos

# 画面内最大サイズの人のidを取得
#target_joint: 0:鼻 1: 心臓
def getBestHumanId(subset, candidate, target_joint =1, log = True):

    best_human_idx =0
    best_human_size=0

    for n in range(len(subset)):                  # subset[n]:n番目の人の関節 一人だけにしたいときはここでn=0
    #writer.writerow(['person'+str(n)])

        d = 0 # 首からの距離計算用

        #首があるか
        if subset[n][target_joint] < 0:# 首がなければやめる 関節がない場合-1が代入されている
            print("target was not found...")
            continue

        for i in range(18): # 関節の数

            index = subset[n][i] 
            index_ne = subset[n][target_joint] 

            if i == target_joint:
                continue
            # 今の関節の座標
            X = candidate[index.astype(int), 0]
            Y = candidate[index.astype(int), 1]
            # 首の座標
            Xne= candidate[index_ne.astype(int),0]
            Yne= candidate[index_ne.astype(int),1]


            #首からの各ポイントへの距離を加算
            #d += math.sqrt((X - Xne)**2 + (Y-Yne)**2)
            d += abs(X - Xne) + abs(Y-Yne)

        if d > best_human_size:
            best_human_idx = n
            best_human_size = d
        if log == True:
            print("person%2d: %f" %(n, d))

    if log == True:
        print("biggest:%d" %best_human_idx) # 4と出力されてほしい
    return best_human_idx

########################################################
# 画面内最大サイズの人のidを取得 # 目の間の距離で判定
# target_joint: 0:鼻 1: 首

# ０：鼻、１：首、２：右肩、3：右肘、４：右手首、５：左肩、
# ６：左肘、７：左手首、８：右腰、９：右膝、10：右足首、
# 11：左腰、12：左膝、13：左足首、14：右目、15：左目、16：右耳、17：左耳

def getBestHumanId_eyes(subset, candidate, target_joint =1, log = True):

    best_human_idx =0
    best_human_size=0

    for n in range(len(subset)):                  # subset[n]:n番目の人の関節 一人だけにしたいときはここでn=0
    #writer.writerow(['person'+str(n)])

        d = 0 # 首からの距離計算用

        #首があるか
        if subset[n][14] < 0 or subset[n][15] < 0: # 目がない場合やめる 関節がない場合-1が代入されている
            print("eyes was not found...")
            continue

        index_right_eye = subset[n][14] 
        index_left_eye  = subset[n][15]

        # 右目の座標
        Xr = candidate[index_right_eye.astype(int), 0]
        Yr = candidate[index_right_eye.astype(int), 1]
        # 左目の座標
        Xl= candidate[index_left_eye.astype(int),0]
        Yl= candidate[index_left_eye.astype(int),1]

        #両目間の距離を計算
        d = math.sqrt((Xr - Xl)**2 + (Yr - Yl)**2)

        if d > best_human_size:
            best_human_idx = n
            best_human_size = d
        if log == True:
            print("person%2d: %f" %(n, d))

    if log == True:
        print("biggest:%d" %best_human_idx) # 4と出力されてほしい
    return best_human_idx

# 辞書に保存する関数 ###############################
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)
# 辞書を読み込む関数################################
def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
###################################################   