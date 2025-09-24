import torch
import gc

def eval_cell_acc():
    '''
    computes the token-level accuracies
    cell_acc:     total token-level accuracy
    cell_acc_101: accuracy for token 101
    etc.
    '''
    model.eval()
    with torch.no_grad():
        l,X,Y = next(iter(dataloader))
        X,Y = X.to(device), Y.bool().to(device)
        out = model(X)
        
        ncells, n_0, n_102, n_101, n_lvl = 0, 0, 0, 0, 0
        correct, correct_0, correct_102, correct_101, correct_lvl = 0, 0, 0, 0, 0
        for j in range(out.shape[0]):
            for t in range(out.shape[1]):
                if Y[j][t].sum() > 0:
                    pred = out[j][t].argmax()
                    if Y[j][t][pred] > 0: correct += 1
                    ncells += 1

                    if Y[j][t][0] > 0:
                        if pred == 0: correct_0 += 1
                        n_0 += 1

                    if Y[j][t][102] > 0:
                        if pred == 102: correct_102 += 1
                        n_102 += 1
                    
                    if Y[j][t][101] > 0:
                        if pred == 101: correct_101 += 1
                        n_101 += 1
                        
                    if Y[j][t][1:20].sum() > 0:
                        if Y[j][t][pred] > 0: correct_lvl += 1
                        n_lvl += 1
                    
        cell_acc = correct/(ncells+1e-6)
        cell_acc_0 = correct_0/(n_0+1e-6)
        cell_acc_102 = correct_102/(n_102+1e-6)
        cell_acc_101 = correct_101/(n_101+1e-6)
        cell_acc_lvl = correct_lvl/(n_lvl+1e-6)

        gc.collect()
        torch.cuda.empty_cache()
    
        return cell_acc, cell_acc_0, cell_acc_102, cell_acc_101, cell_acc_lvl

def edit_preds(Xin):
    '''
    given a generated sequence Xin, finds the final generated solution path (for each cell, keeps the last assigned move)
    '''
    preds = Xin.long().squeeze().cpu().tolist()
    xx = np.zeros(100)
    for i in preds:
        if (i > 110) and (i < 1000):
            rc = i//10
            xx[rc] = i%10

    preds_edited = [int(i*10+xx[i]) for i in range(len(xx)) if xx[i] > 0]
    return preds_edited

def eval_board_acc(dset='random'):
    '''
    computes board accuracy for a batch of inputs
    '''
    model.eval()
    correct = 0
    with torch.no_grad():
        # Random
        if dset == 'random':
            _,X,_ = next(iter(dataloader_test))

        # Kaggle 3m
        if dset == 'kaggle_3m':
            X = []
            for jj in range(bs_test):
                idx = np.random.randint(0,len(K))
                s = K['puzzle'].iloc[idx]
                l = 81 - s.count('.')
                s = s.replace('.','0')
                slist0 = [int(ii) for ii in s]
        
                s = K['solution'].iloc[idx]
                s = s.replace('.','0')
                slist = [int(ii) for ii in s]
                
                path0 = []
                for ii in range(81):
                    row = ii//9 + 1
                    col = ii%9 + 1
                    if slist0[ii] > 0: path0 += [row,col,slist0[ii]]
        
                X0 = [100*row[0]+10*row[1]+row[2] for row in np.array(path0).reshape(-1,3)]
        
                path = []
                for ii in range(81):
                    row = ii//9 + 1
                    col = ii%9 + 1
                    if slist[ii] > 0: path += [row,col,slist[ii]]
        
                Xall_ = [100*row[0]+10*row[1]+row[2] for row in np.array(path).reshape(-1,3)]
        
                X_sol = [i for i in Xall_ if not (i in X0)]
                Xi = X0 + [100] + X_sol
                X.append(torch.tensor(Xi))

            X = torch.stack(X)


        l100 = (X == 100).float().argmax(dim=1)  # shape: [batch_size]
        ptrs = l100.clone()
        l100_max = l100.max()

        Xin = X[:,:l100_max+1].clone()
        for j in range(Xin.shape[0]):
            Xin[j,l100[j]+1:] = 1000
        
        finished = torch.zeros(Xin.shape[0])
        Xf, Gt = [], []
        Xclone = X.clone()
        while True:
            out = model(Xin.long().to(device))
            pred = out.argmax(-1)
            next_tokens = pred[torch.arange(Xin.shape[0]), ptrs]
            
            Xin = torch.concatenate((Xin, 1000*torch.ones(Xin.shape[0],1)), dim=-1)
            Xin[torch.arange(Xin.shape[0]), ptrs+1] = next_tokens.to(Xin.dtype).cpu()
        
            finished[next_tokens==0] = 1

            for j in range(len(finished)):
                if finished[j]==1: 
                    Xf.append(Xin[j])
                    Gt.append(Xclone[j])

            mask = (finished==0).bool()
            Xin = Xin[mask]
            Xclone = Xclone[mask]
            ptrs = ptrs[mask]
            finished = finished[mask]
            
            ptrs += 1

            if len(finished) == 0: break
            if Xin.shape[-1] > 500: break  # maximum-allowed length for generated sequence

        for j in range(len(finished)):
            Xf.append(Xin[j])
            Gt.append(Xclone[j])
            
        for jj in range(len(Xf)):
            result = np.sum(edit_preds(Xf[jj]) == edit_preds(Gt[jj]))
            correct += result
    
        board_accx = correct/X.shape[0]
    
        gc.collect()
        torch.cuda.empty_cache()
        del Xin, Xclone, out, X
        return board_accx