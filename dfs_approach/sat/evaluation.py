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
                        
                    if Y[j][t][200:].sum() > 0:
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


def get_board_acc(X):
    '''
    checks board accuracy for a given 1-in-3 SAT instance's solution path
    '''
    preds = X.long().squeeze().cpu().tolist()
    l100 = preds.index(100)
    clauses = np.array(preds[:l100]).reshape(-1,3)
    
    # get solution found
    Nn = Nvar
    var_values = np.zeros(Nn+1)
    for i in range(l100+1,len(preds)):  #last observed instance of each variable will be kept
        if preds[i]==0: break
        if preds[i] <= 2*Nn: #if is about the value of a variable (i.e. not a special token)
            posi = preds[i] if preds[i]<=Nn else preds[i]-Nn
            var_values[posi] = 1 if preds[i]<=Nn else 0

    # check if solution satisfies all given clauses (only 1 variable should be True in every clause)
    all_correct = 1
    for clausei in clauses:
        si = 0
        for vari in clausei: 
            result = 0
            posi = vari if vari<=Nn else vari-Nn
            if (vari <= Nn) and (var_values[posi] > 0): result = 1 
            if (vari > Nn) and (var_values[posi] == 0): result = 1 
            si += result
        if si != 1: 
            all_correct = 0
            break
    
    return all_correct


def eval_board_acc():

    model.eval()
    correct = 0
    with torch.no_grad():
        _,X,_ = next(iter(dataloader_test))

        l100 = (X == 100).float().argmax(dim=1)  # shape: [batch_size]
        ptrs = l100.clone()
        l100_max = l100.max()

        Xin = X[:,:l100_max+1].clone()
        for j in range(Xin.shape[0]):
            Xin[j,l100[j]+1:] = 1000
        
        finished = torch.zeros(Xin.shape[0])
        Xf = []
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

            #continue with only unfinished sequences
            mask = (finished==0).bool()
            Xin = Xin[mask]
            Xclone = Xclone[mask]
            ptrs = ptrs[mask]
            finished = finished[mask] 
            
            ptrs += 1

            if len(finished) == 0: break
            if Xin.shape[-1] > 2*max_len: break # maximum-allowed length for generated sequence (user-defined)

        # add last unfinished sequences
        for j in range(len(finished)):
            Xf.append(Xin[j])

        # get board accuracy for each instance
        for jj in range(len(Xf)):
            result = get_board_acc(Xf[jj])
            correct += result
    
        board_accx = correct/X.shape[0]
    
        gc.collect()
        torch.cuda.empty_cache()
        del Xin, Xclone, out, X
        
        return board_accx
