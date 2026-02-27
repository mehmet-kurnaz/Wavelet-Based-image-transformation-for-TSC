#libraries
#from skimage.io import imsave
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import prep_data, dim_reduce, get_score
from time import time
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import copy
#from clf_repo import clf_dict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#from quant import Quant
import matplotlib.pyplot as plt
import os
from scipy.stats import mode
import pywt
from scipy import signal
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from PIL import Image
import os
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.callbacks import ModelCheckpoint
import gc
from tensorflow.keras import backend as K
#parameters

mode_ = 'full'
save_df = 0
load_feat = False
pc_name = 'asus'
compute_durs = 1
compute_perf = 1
save_df_perf = 1
save_df_acc = 1 #accuracy?
n_re_ = 30
warnings.filterwarnings("ignore")

clfnames = ['quant']  #denenecek classifier listesi
clf_name = 'etc'

use_scaler = False
dim_reduction = False

scaler = StandardScaler(with_mean=False)

if mode_ == 'trial' or (compute_durs and not compute_perf):
    n_re = 3
else:
    n_re = 30

eval_metric = 'acc'
rseed = None

df_metadata = pd.read_excel("C:/Users/alagozc/Desktop/tsc/dsets_ucr_ts_uv/ucr_dsets_all_metadata.xlsx")
df_ref_full_path = 'C:/Users/alagozc/Desktop/tsc/dsets_ucr_ts_uv/_benchmark_results/FreshPRINCE_accuracy.csv'
df_ref = pd.read_csv(df_ref_full_path)

ind_ref_path = "C:/Users/alagozc/Desktop/tsc/dsets_ucr_ts_uv/PythonResampleIndices"

for clf_name in clfnames:

    # Prepare results tables
    df_full_path = f'results/perf_{clf_name}_{n_re_}r.csv'
    df_acc_full_path = f'results/{clf_name}_acc_{n_re_}r.csv'
    # df_bac_full_path=f'results/all_set/drstf_{ver_}_bac_{clf_name}_{n_re}r_{pc_name}.csv'
    # df_auc_full_path=f'results/all_set/drstf_{ver_}_auc_{clf_name}_{n_re}r_{pc_name}.csv'
    # df_nll_full_path=f'results/all_set/drstf_{ver_}_nll_{clf_name}_{n_re}r_{pc_name}.csv'
    # df_f1_full_path=f'results/all_set/drstf_{ver_}_f1_{clf_name}_{n_re}r_{pc_name}.csv'
    # df_dur_full_path=f'results/all_set/drstf_{ver_}_dur_{clf_name}_{n_re}r_{pc_name}.csv'

    if mode_ == 'partial':
        df_ = pd.read_csv(df_full_path)
        df_acc = pd.read_csv(df_acc_full_path)
        # df_bac = pd.read_csv(df_bac_full_path)
        # df_auc = pd.read_csv(df_auc_full_path)
        # df_nll = pd.read_csv(df_nll_full_path)
        # df_f1 = pd.read_csv(df_f1_full_path)
        # df_dur = pd.read_csv(df_dur_full_path)
    else:
        df_ = pd.DataFrame({'dataset': [],
                            'n_class': [],
                            'mean_acc': [],
                            'mean_bac': [],
                            'mean_auc': [],
                            'mean_nll': [],
                            'mean_f1': [],
                            f'dur_tr_{pc_name}': [],
                            f'dur_te_{pc_name}': []})
        df_acc = pd.DataFrame({'dataset': [], 'n_class': [], **{str(i): [] for i in range(n_re)}, 'mean': []})
        # df_bac = pd.DataFrame({'dataset':[],'n_class':[], **{str(i):[] for i in range(n_re)},'mean':[]})
        # df_auc = pd.DataFrame({'dataset':[],'n_class':[], **{str(i):[] for i in range(n_re)},'mean':[]})
        # df_nll = pd.DataFrame({'dataset':[],'n_class':[], **{str(i):[] for i in range(n_re)},'mean':[]})
        # df_f1 = pd.DataFrame({'dataset':[],'n_class':[], **{str(i):[] for i in range(n_re)},'mean':[]})
        # df_dur = pd.DataFrame({'dataset':[],'dur_trans_tr':[],'dur_fit':[],'dur_trans_te':[],'dur_pred':[]})

    # %% run through datasets
    if mode_ == 'trial':
        start_id = 11
        stop_id = start_id + 3
        save_df = 0
    elif mode_ == 'full':
        start_id = 0
        stop_id = None
    elif mode_ == 'partial':
        start_id = 0
        stop_id = None
    t_total = time()
    mean_all = []
    #for dset_name in df_ref['Resamples:'][start_id:stop_id]:
    for dset_name in df_ref['Resamples:'][121:stop_id]:
        #dset_name = "ACSF1"

        i_dset = df_ref.index[df_ref['Resamples:'] == dset_name][0]

        dset, _ = prep_data(dset_name,
                            repo='ucr_drive',
                            orig_split=True, )
        x_train, x_test, y_train, y_test = dset  #split burdaki gibi mi?
        X = np.r_[x_train, x_test]
        y = np.r_[y_train, y_test]
        classes, class_counts = np.unique(y_train, return_counts=True)
        n_class = len(classes)

        df_.at[i_dset, 'dataset'] = dset_name
        df_.at[i_dset, 'n_class'] = n_class

        df_acc.at[i_dset, 'dataset'] = dset_name
        df_acc.at[i_dset, 'n_class'] = n_class

        n_sample_train, len_ = x_train.shape
        n_sample_test = x_test.shape[0]
        n_sample = n_sample_train + n_sample_test
        print(
            f'--- [{i_dset}] {dset_name}, {n_class}-class, length:{len_}, n_tr:{n_sample_train}, n_te:{n_sample_test}, {clf_name} ---',
            end='')

        n_samples = X.shape[0]
        test_size = len(y_test) / (len(y_test) + len(y_train))

        accs = []
        bacs = []
        f1s = []
        aucs = []
        nlls = []
        # durs_trans_tr = []
        # durs_trans_te = []
        durs_fit = []
        durs_pred = []

        for fold_ in range(n_re):
            # if fold_==0:
            #     x_tr_trans, x_te_trans, y_tr, y_te = dset_trans
            # else:
            #     x_tr_trans, x_te_trans, y_tr, y_te = train_test_split(X_trans, y, test_size=test_size, random_state=fold_, stratify=y)

            idx_tr = np.loadtxt(f'{ind_ref_path}/{dset_name}/resample{fold_}Indices_TRAIN.txt', dtype=int)
            idx_te = np.loadtxt(f'{ind_ref_path}/{dset_name}/resample{fold_}Indices_TEST.txt', dtype=int)
            x_tr = X[idx_tr]
            y_tr = y[idx_tr]
            x_te = X[idx_te]
            y_te = y[idx_te]

            n_timestamps = x_tr.shape[1]
            if n_timestamps < 64:
                tmstep= n_timestamps
            elif n_timestamps < 300:
                tmstep= 64
            elif n_timestamps < 600:
                tmstep= 128
            else:
                tmstep= 256
            gasf = GramianAngularField(image_size=tmstep, method='summation')
            mtf = MarkovTransitionField(image_size=tmstep, n_bins=8)

            rp = RecurrencePlot(threshold='point', percentage=20)
            X_rp_train, X_gasf_train,X_mtf_train = [],[],[]
            if True:
                for sig in x_tr:
                    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
                    sig_interp = np.interp(np.linspace(0, len(sig) - 1, tmstep), np.arange(len(sig)), sig)
                
                    # RP
                    rp_img = rp.fit_transform(sig_interp.reshape(1, -1))[0]
                    X_rp_train.append(rp_img)
                
                    # GASF
                    gaf_img = gasf.fit_transform(sig_interp.reshape(1, -1))[0]
                    X_gasf_train.append(gaf_img)
                
                    # MTF
                    mtf_img = mtf.fit_transform(sig_interp.reshape(1, -1))[0]
                    X_mtf_train.append(mtf_img)
                
                X_rp_train = np.array(X_rp_train, dtype=np.float32)
                X_gasf_train = np.array(X_gasf_train, dtype=np.float32)
                X_mtf_train = np.array(X_mtf_train, dtype=np.float32)
                
                # Test verisi dönüşümü
                X_rp_test, X_gasf_test, X_mtf_test = [], [], []
                
                for sig in x_te:
                    # Normalize + interpolate
                    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
                    sig_interp = np.interp(np.linspace(0, len(sig) - 1, tmstep), np.arange(len(sig)), sig)
                
                    # RP
                    rp_img = rp.transform(sig_interp.reshape(1, -1))[0]
                    X_rp_test.append(rp_img)
                
                    # GASF
                    gaf_img = gasf.transform(sig_interp.reshape(1, -1))[0]
                    X_gasf_test.append(gaf_img)
                
                    # MTF
                    mtf_img = mtf.transform(sig_interp.reshape(1, -1))[0]
                    X_mtf_test.append(mtf_img)
                
                X_rp_test = np.array(X_rp_test, dtype=np.float32)
                X_gasf_test = np.array(X_gasf_test, dtype=np.float32)
                X_mtf_test = np.array(X_mtf_test, dtype=np.float32)


                X_train_img = np.stack((X_gasf_train, X_mtf_train, X_rp_train), axis=-1)
                X_test_img = np.stack((X_gasf_test, X_mtf_test, X_rp_test), axis=-1)


                X_train_img = np.stack((X_gasf_train, X_mtf_train, X_rp_train), axis=-1)
                X_test_img = np.stack((X_gasf_test, X_mtf_test, X_rp_test), axis=-1)

                save_dir = os.path.join(ind_ref_path, dset_name, f"resample{fold_}")
                os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
                os.makedirs(os.path.join(save_dir, "test"), exist_ok=True)

                # Train kaydet


                #for i in range(len(X_train_img)):
                #    filename = os.path.join(save_dir, "train", f"{i}_label{y_tr[i]}.png")
                #    img_to_save = (X_train_img[i] * 255).astype(np.uint8)
                #    imsave(filename, img_to_save)

                # Test kaydet
                #for i in range(len(X_test_img)):
                #    filename = os.path.join(save_dir, "test", f"{i}_label{y_te[i]}.png")
                #    img_to_save = (X_test_img[i] * 255).astype(np.uint8)
                #    imsave(filename, img_to_save)

                #print(f"Resample {fold_} -> train ve test resimleri kaydedildi.")

            def transform_dataset(X):
                n_timestamps = X.shape[1]
                X_img = []
                for signal in X:
                    coeffs = pywt.wavedec(signal, 'db4', level=2)
                    channels = []

                    c = np.ravel(coeffs[1])
                    c = (c - np.min(c)) / (np.max(c) - np.min(c) + 1e-6)
                    c_resized = np.interp(np.linspace(0, len(c) - 1, n_timestamps), np.arange(len(c)), c)
                    gaf_img = gasf.fit_transform(c_resized.reshape(1, -1))[0]


                    c = np.ravel(coeffs[0])
                    c = (c - np.min(c)) / (np.max(c) - np.min(c) + 1e-6)
                    c_resized = np.interp(np.linspace(0, len(c) - 1, n_timestamps), np.arange(len(c)), c)
                    mtf_img = mtf.fit_transform(c_resized.reshape(1, -1))[0]


                    c = np.ravel(coeffs[2])
                    c = (c - np.min(c)) / (np.max(c) - np.min(c) + 1e-6)
                    c_resized = np.interp(np.linspace(0, len(c) - 1, n_timestamps), np.arange(len(c)), c)
                    rp_img = rp.fit_transform(c_resized.reshape(1, -1))[0]
                    rp_img = resize(rp_img, (tmstep, tmstep), mode='reflect', anti_aliasing=True)

                    # Üç kanalı birleştir
                    channels.append(gaf_img)
                    channels.append(rp_img)
                    channels.append(mtf_img)
                    X_img.append(np.stack(channels, axis=-1))

                return np.array(X_img)


            # Kullanımı

            #X_train_img = transform_dataset(x_tr)
            #X_test_img = transform_dataset(x_te)

            print(X_train_img.shape)
            print(X_test_img.shape)

            #X_train_img_df = tf.data.Dataset.from_tensor_slices((X_train_img, y_tr)).batch(16)
            #X_test_img_df = tf.data.Dataset.from_tensor_slices((X_test_img, y_te)).batch(16)
            model = models.Sequential([
                layers.Conv2D(16, (3, 3), activation='relu', input_shape=(tmstep, tmstep, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
            ])
            if n_class == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss_fn = 'binary_crossentropy'
            else:
                model.add(layers.Dense(n_class, activation='softmax'))
                # y one-hot ise categorical, integer ise sparse_categorical
                loss_fn = 'categorical_crossentropy'

            model.compile(optimizer='adam',
                          loss=loss_fn,
                          metrics=['accuracy'])
            # Eğitim
            t0 = time()
            
            checkpoint_path = f"best_model_fold_{fold_}.h5"
            checkpoint = ModelCheckpoint(
                checkpoint_path, monitor='val_accuracy', verbose=0,
                save_best_only=True, mode='max'
            )
            if n_class == 2:
                # Binary classification
                # Etiketleri olduğu gibi bırak (0 / 1)
                y_tr_oh = y_tr
                y_te_oh = y_te
            else:
                y_tr_oh = to_categorical(y_tr, num_classes=n_class)
                y_te_oh = to_categorical(y_te, num_classes=n_class)

            history = model.fit(X_train_img, y_tr_oh,
                                validation_data=(X_test_img, y_te_oh),
                                epochs=10,
                                verbose=0,
                                batch_size=16,
                                callbacks=[checkpoint]
                                )
            dur_fit = time() - t0
            model.load_weights(checkpoint_path)

            # Tahmin
            t0 = time()
            y_pred_proba = model.predict(X_test_img)
            dur_pred = time() - t0
            #clf = copy.deepcopy(clf_dict[clf_name])

            if n_class == 2:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()

                acc_ = metrics.accuracy_score(y_te, y_pred)
                bac_ = metrics.balanced_accuracy_score(y_te, y_pred)
                f1_ = metrics.f1_score(y_te, y_pred)
                auc_ = metrics.roc_auc_score(y_te, y_pred_proba.ravel())
                nll_ = metrics.log_loss(y_te, y_pred_proba.ravel())
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
                acc_ = metrics.accuracy_score(y_te, y_pred)
                bac_ = metrics.balanced_accuracy_score(y_te, y_pred)
                f1_ = metrics.f1_score(y_te, y_pred, average="weighted")
                auc_ = metrics.roc_auc_score(y_te, y_pred_proba, multi_class="ovr")
                nll_ = metrics.log_loss(y_te, y_pred_proba)
            #y_pred_proba = clf.predict_proba(x_te)
            K.clear_session()
            gc.collect()
            del model
            accs.append(acc_)
            bacs.append(bac_)
            f1s.append(f1_)
            aucs.append(auc_)
            nlls.append(nll_)

            durs_fit.append(dur_fit)
            durs_pred.append(dur_pred)

            df_acc.at[i_dset, str(fold_)] = acc_

            # print(f'fold:{fold_} dur_proto:{dur_poroto:.2f}, dur_fit:{dur:.2f}, acc:{acc_:.4f}')

        mean_acc = np.mean(accs)
        mean_all.append(mean_acc)
        running_mean_all = np.mean(mean_all)
        df_acc.at[i_dset, 'mean'] = mean_acc
        acc_mean_2 = df_[df_['n_class'] < 3]['mean_acc'].mean()

        if compute_perf:
            mean_bac = np.mean(bacs)
            mean_auc = np.mean(aucs)
            mean_nll = np.mean(nlls)
            mean_f1 = np.mean(f1s)

            df_.at[i_dset, 'mean_acc'] = mean_acc
            df_.at[i_dset, 'mean_bac'] = mean_bac
            df_.at[i_dset, 'mean_auc'] = mean_auc
            df_.at[i_dset, 'mean_nll'] = mean_nll
            df_.at[i_dset, 'mean_f1'] = mean_f1

        if compute_durs:
            dur_fit = np.median(durs_fit)
            dur_pred = np.median(durs_pred)

            df_.at[i_dset, f'dur_tr_{pc_name}'] = dur_fit
            df_.at[i_dset, f'dur_te_{pc_name}'] = dur_pred

        acc_mean_2 = df_[df_['n_class'] < 3]['mean_acc'].mean()
        #print(f'mean acc:{acc_mean:.4f}')
        # print(f'Running mean:{running_mean_all:.4f}\n')
        print(f'Running mean 2-class:{acc_mean_2:.4f} all:{running_mean_all:.4f}')

        if save_df_perf: df_.to_csv(df_full_path, index=False)
        if save_df_acc: df_acc.to_csv(df_acc_full_path, index=False)

    # %%
    if compute_perf:
        df_.at['summary', 'mean_acc'] = df_['mean_acc'].mean()
        df_.at['summary', 'mean_bac'] = df_['mean_bac'].mean()
        df_.at['summary', 'mean_auc'] = df_['mean_auc'].mean()
        df_.at['summary', 'mean_nll'] = df_['mean_nll'].mean()
        df_.at['summary', 'mean_f1'] = df_['mean_f1'].mean()

    if compute_durs:
        df_.at[df_.last_valid_index(), f'dur_tr_{pc_name}'] = df_[f'dur_tr_{pc_name}'].sum()
        df_.at[df_.last_valid_index(), f'dur_te_{pc_name}'] = df_[f'dur_te_{pc_name}'].sum()

    dur_total = time() - t_total
    print(f'total dur:{dur_total:.1f} secs')

    if save_df_perf: df_.to_csv(df_full_path, index=False)
    if save_df_acc: df_acc.to_csv(df_acc_full_path, index=False)