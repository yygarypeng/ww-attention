import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler

def load_particles_from_h5(filename):
    result = {}

    with h5py.File(filename, "r") as f:
        # For each category (ggF_train, ggF_test, VBF_train, etc.)
        for category_name in f.keys():
            category_data = {}
            
            # For each particle/object group within the category
            for group_name in f[category_name].keys():
                group_data = {}

                # Load datasets (numpy arrays)
                if isinstance(f[category_name][group_name], h5py.Group):
                    for dataset_name in f[category_name][group_name].keys():
                        group_data[dataset_name] = f[category_name][group_name][dataset_name][:]

                    # Load attributes (scalars)
                    for attr_name, attr_value in f[category_name][group_name].attrs.items():
                        group_data[attr_name] = attr_value
                else:
                    # Handle case where it's a dataset directly
                    group_data = f[category_name][group_name][:]

                category_data[group_name] = group_data

            result[category_name] = category_data

    return result

def load_data(data_path):
    
    data = load_particles_from_h5(data_path)

    def col(a):
        return a.reshape(a.shape[0], -1)
    
    def deta(eta1, eta2):
        return np.abs(eta1 - eta2)

    def dphi_pi(phi1, phi2):
        phi_diff = phi1 - phi2
        phi_diff = np.where(phi_diff < 0.0, -phi_diff, phi_diff)
        phi_diff = np.where(phi_diff > 2.0 * np.pi, phi_diff - 2.0 * np.pi, phi_diff)
        phi_diff = np.where(phi_diff >= np.pi, 2.0 * np.pi - phi_diff, phi_diff)
        return np.divide(phi_diff, np.pi)

    # Collect all training and target objects from all categories
    all_train_objs = []
    all_target_objs = []
    
    # Iterate through all categories (ggF_train, ggF_test, VBF_train, etc.)
    for category in data.keys():
        category_data = data[category]
        
        # training features
        lep_pos_px = category_data["pos_lep"]["px"]
        lep_pos_py = category_data["pos_lep"]["py"]
        lep_pos_pz = category_data["pos_lep"]["pz"]
        lep_pos_energy = category_data["pos_lep"]["energy"]
        lep_neg_px = category_data["neg_lep"]["px"]
        lep_neg_py = category_data["neg_lep"]["py"]
        lep_neg_pz = category_data["neg_lep"]["pz"]
        lep_neg_energy = category_data["neg_lep"]["energy"]
        
        lep_pos_pt = category_data["pos_lep"]["pt"]
        lep_neg_pt = category_data["neg_lep"]["pt"]
        lep_pos_eta = category_data["pos_lep"]["eta"]
        lep_neg_eta = category_data["neg_lep"]["eta"]
        lep_pos_phi = category_data["pos_lep"]["phi"]
        lep_neg_phi = category_data["neg_lep"]["phi"]

        met_px = category_data["met"]["px"]
        met_py = category_data["met"]["py"]
        met_pt = category_data["met"]["pt"]
        met_phi = category_data["met"]["phi"]
        
        dphi_l1met = dphi_pi(lep_pos_phi, met_phi)
        dphi_l2met = dphi_pi(lep_neg_phi, met_phi)
        dphi_l1l2 = dphi_pi(lep_pos_phi, lep_neg_phi)
        deta_l1l2 = deta(lep_pos_eta, lep_neg_eta)  
        
        # only select first 3 jets (leading/subleading/subsubleading)
        jet_px = category_data["jets"]["px"][:, 0:3]
        jet_py = category_data["jets"]["py"][:, 0:3]
        jet_pz = category_data["jets"]["pz"][:, 0:3]
        jet_energy = category_data["jets"]["energy"][:, 0:3]
        jet_btag = category_data["jets"]["btag"][:, 0:3]
        n_jets = category_data["jets"]["n_jets"]
        n_bjets = category_data["jets"]["n_bjets"]

        # pack them
        # all training mass-like objects are in GeV unit
        
        train_obj = np.concatenate([
            col(lep_pos_px),
            col(lep_pos_py),
            col(lep_pos_pz),
            col(lep_pos_energy),
            col(lep_neg_px),
            col(lep_neg_py),
            col(lep_neg_pz),
            col(lep_neg_energy),
            col(jet_px[:, 0]),
            col(jet_py[:, 0]),
            col(jet_pz[:, 0]),
            col(jet_energy[:, 0]),
            col(jet_px[:, 1]),
            col(jet_py[:, 1]),
            col(jet_pz[:, 1]),
            col(jet_energy[:, 1]),
            col(jet_px[:, 2]),
            col(jet_py[:, 2]),
            col(jet_pz[:, 2]),
            col(jet_energy[:, 2]),
            col(met_px),
            col(met_py),
            # col(lep_pos_pt),
            # col(lep_neg_pt),
            # col(lep_pos_eta),
            # col(lep_neg_eta),
            # col(lep_pos_phi),
            # col(lep_neg_phi),
            # dphi has normalized to pi; ie. range [0, 1]
            # col(met_pt),
            # col(met_phi),
            col(dphi_l1met), # l1 -> pos_lep; l2 -> neg_lep 
            col(dphi_l2met),
            col(dphi_l1l2),
            # deta has absolute value; ie, range [0, inf)
            col(deta_l1l2), # 14
            # col(jet_btag),# check definitin!!
            # col(n_jets),
            # col(n_bjets),
        ], axis=-1)
        
        # target objects
        target_obj = np.concatenate([
			col(category_data["truth_pos_w"]["px"]),
			col(category_data["truth_pos_w"]["py"]),
			col(category_data["truth_pos_w"]["pz"]),
			col(category_data["truth_pos_w"]["energy"]),
			col(category_data["truth_neg_w"]["px"]),
			col(category_data["truth_neg_w"]["py"]),
			col(category_data["truth_neg_w"]["pz"]),
			col(category_data["truth_neg_w"]["energy"]),
			col(category_data["truth_pos_w"]["m"]),
			col(category_data["truth_neg_w"]["m"]),
        ], axis=-1)
        
        all_train_objs.append(train_obj)
        all_target_objs.append(target_obj)
    
    # Concatenate all categories
    train_obj = np.concatenate(all_train_objs, axis=0)
    target_obj = np.concatenate(all_target_objs, axis=0)
    
    print("Training objects shape:", train_obj.shape)
    print("Target objects shape:", target_obj.shape)

    # Remove rows with NaN or infinite values
    valid_train = np.isfinite(train_obj).all(axis=1)
    valid_target = np.isfinite(target_obj).all(axis=1)
    valid_idx = valid_train & valid_target

    train_obj = train_obj[valid_idx]
    target_obj = target_obj[valid_idx]

    print("Removed", (~valid_idx).sum(), "rows with NaN or infinite values")
    
    _ = StandardScaler().fit_transform(train_obj)
    std_mean_train, std_scale_train = StandardScaler().fit(train_obj).mean_, StandardScaler().fit(train_obj).scale_
    _ = StandardScaler().fit_transform(target_obj)
    std_mean_target, std_scale_target = StandardScaler().fit(target_obj).mean_, StandardScaler().fit(target_obj).scale_


    return train_obj, target_obj, (std_mean_train, std_scale_train), (std_mean_target, std_scale_target)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import torch
    data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5"
    train_obj, target_obj, _, _ = load_data(data_path)
    w_pos_mass = target_obj[:, 8]
    w_neg_mass = target_obj[:, 9]
    plt.hist(w_pos_mass, bins=50, range=(0, 120), histtype='step', label='W+ mass')
    plt.hist(w_neg_mass, bins=50, range=(0, 120), histtype='bar', label='W- mass')
    plt.xlabel("W mass [GeV]")
    plt.ylabel("Entries")
    plt.legend()
    train_obj = torch.tensor(train_obj)
    print(torch.abs(train_obj[:, 8:12][:5]).sum(dim=1) == 0)
    print(torch.abs(train_obj[:, 12:16][:5]).sum(dim=1) == 0)
    print(torch.abs(train_obj[:, 16:20][:5]).sum(dim=1) == 0)
    print(train_obj[:, 12:16][:5])
    print(train_obj[:, 16:20][:5])