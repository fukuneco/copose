# 類似度を求める損失関数 

import math
import numpy as np
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

#コサイン類似度を取得########################################
# @v_l: pickleから取得した関節座標
# @v_y: 使用していた画像の関節座標
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #np.dot()内積
############################################################

############################################################
# @v_l: 事前評価ファイルの座標18個(x,y)
# @v_y: 基準ファイルの座標18個(x,y)
# ex. v_l[1][0]: 首(id:1)のx座標
# @neck_x: 左右反転基準となる座標(x)
# return d    : 距離
#        d_inv: 基準ファイルのベクトルをx反転した場合の距離
def getCosSimilarity(v_l, v_y, neck_x):
    
    class Vec:
        def __init__(self):
            self.x0 = None
            self.x1 = None
            self.y0 = None
            self.y1 = None
            
    d = 0
    d_inv= 0 #首で反転した場合
    
    cnt = 0 # 後で平均を取る
    vec_l = Vec()
    vec_y = Vec()
    #print(vec_l.__dict__.values())
    #print(np.nan() in vec_l.__dict__.values())

    for i in range(17): # 関節間17個のベクトル

        vec_l.x0 = v_l[np.array(limbSeq[i][0]-1)][0]
        vec_l.y0 = v_l[np.array(limbSeq[i][0]-1)][1]
        vec_l.x1 = v_l[np.array(limbSeq[i][1]-1)][0]
        vec_l.y1 = v_l[np.array(limbSeq[i][1]-1)][1]    
        
        vec_y.x0 = v_y[np.array(limbSeq[i][0]-1)][0]
        vec_y.y0 = v_y[np.array(limbSeq[i][0]-1)][1]
        vec_y.x1 = v_y[np.array(limbSeq[i][1]-1)][0]
        vec_y.y1 = v_y[np.array(limbSeq[i][1]-1)][1] 
        
        # nan判定を入れたい
        #print(vec_l.__dict__.values())
        #print(vec_y.__dict__.values())
        #print("-----")
        nan_chk_l = np.array([vec_l.x0, vec_l.y0, vec_l.x1, vec_l.y1])
        nan_chk_y = np.array([vec_y.x0, vec_y.y0, vec_y.x1, vec_y.y1])
        
        if np.isnan(nan_chk_y).any(): # your fileに関節ベクトルがない場合
            #print("***************************** Nan is found... %2d" %i) 
            #flg_first_NJ_your_joints = True
            continue
            
        if np.isnan(nan_chk_l).any(): # pickleで読み込んだファイルの関節ベクトルがない場合
            print("***************************** Nan is found... %2d" %i)
            continue
                
        vl = [vec_l.x1 - vec_l.x0, vec_l.y1 - vec_l.y0]
        vy = [vec_y.x1 - vec_y.x0, vec_y.y1 - vec_y.y0]
        
        # ロードした画像のベクトルをneck_x中心で反転させる
        vl_inv = [neck_x-(vec_l.x1 - vec_l.x0), vec_l.y1 - vec_l.y0] # loaded data
        
        #vy_inv = [vec_y.x1 - vec_y.x0, vec_y.y1 - vec_y.y0]          # your data
        
        # 類似度を加算
        d += cos_sim(vl, vy) # 同じベクトルでない限り1は超えていなそう

        d_inv += cos_sim(vl_inv, vy)
        cnt += 1
    if neck_x < 0: #首がない場合-1が入っている
        d_inv = 9999
        
    #return (d/cnt), (d_inv/cnt)
    return d, d_inv                      # 平均しなくてよいのでは？

#####################################