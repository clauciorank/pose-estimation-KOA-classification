import sys, numpy as np, torch          
sys.path.insert(0, ".")
from src.models.dataset import load_sequence_dataset,compute_class_weights                       
from src.models.lstm_model import build_lstm                            
from src.models.train_lstm import train_model, TrainConfig              
from sklearn.model_selection import StratifiedGroupKFold                
                                                                          
print("=== PROBE Task B (KOA Staging) — 1 fold, 200 epochs ===")        
seq = load_sequence_dataset(filter_groups=["KOA"], label_mode="stage")  
unique_subjs = np.unique(seq.groups)                                    
subj_lbl = {s: int(np.bincount(seq.y[seq.groups==s]).argmax()) for s in 
unique_subjs}                                                           
subj_arr = np.array(unique_subjs)                                       
subj_lbl_arr = np.array([subj_lbl[s] for s in unique_subjs])            
                                                                      
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  
tr_si, te_si = next(iter(sgkf.split(subj_arr, subj_lbl_arr,             
groups=subj_arr)))                                    
tr_subjs = set(subj_arr[tr_si])                                         
tr_mask = np.isin(seq.groups, list(tr_subjs))         
X_tr, y_tr = seq.X[tr_mask], seq.y[tr_mask]                             
print(f"Treino: {len(tr_subjs)} sujeitos, {X_tr.shape[0]} ciclos")
                                                                      
cw = compute_class_weights(y_tr)                      
torch.manual_seed(43)                                                   
model = build_lstm(input_size=4, hidden_size=64, num_layers=2,          
num_classes=3, dropout=0.3)                                             
cfg = TrainConfig(epochs=200, batch_size=32, lr=1e-3,                   
                early_stop=False, verbose=True, log_every=10)         
train_model(model, X_tr, y_tr, X_tr, y_tr, class_weights=cw, cfg=cfg)   
print("=== FIM ===") 
