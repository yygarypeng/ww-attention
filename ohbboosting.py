import os
import multiprocessing
import numpy as np
from ROOT import TLorentzVector, TVector3

class Booster:
    def __init__(self, particles):
        """
        Initializes the Booster with particle 4-vectors in the lab frames.
        The 4-vector define is [px, py, pz, energy] following ROOT's TLorentzVector convention.
        The particle ordering should be: [W+ boson, W+ lepton, W- boson, W- lepton].
        :param particles: np.array of [W0, l0, W1, l1] 4-vectors (px, py, pz, energy) with shape (N, 4)
        """
        print("Make sure the input particles are ordered as [W+, l+, W-, l-]!")
        assert type(particles) == np.ndarray, "Particles should be a numpy array."
        self.particles = particles

    def _boost_to_rest_frame(self, vecs, boost_vec):
        """
        Boosts all TLorentzVectors in vecs by -boost_vec.
        :param vecs: list of TLorentzVector objects
        :param boost_vec: TVector3 representing the boost vector
        """
        for vec in vecs:
            vec.Boost(-boost_vec)

    def _construct_basis(self, WpBoson, Beam_p):
        """
        Constructs orthogonal basis (k, r, n) in Higgs rest frame.
        :param WpBoson: TLorentzVector of W+ boson (we use W+ as k direction)
        :param Beam_p: TLorentzVector of beam direction
        :return: TVector3 k, r, n (orthogonal unit vectors)
        """
        k = TVector3(WpBoson.X(), WpBoson.Y(), WpBoson.Z()).Unit()
        p = TVector3(Beam_p.X(), Beam_p.Y(), Beam_p.Z()).Unit()
        y = p.Dot(k)
        r_length = np.sqrt(1 - y * y)
        r = (1 / r_length) * (p - y * k)
        n = (1 / r_length) * (p.Cross(k))
        return k, r, n

    def _map_to_basis(self, lepton, n, r, k):
        """
        Maps lepton momentum to (n, r, k) basis.
        :param lepton: TLorentzVector of the lepton
        :param n: TVector3 of n basis vector
        :param r: TVector3 of r basis vector
        :param k: TVector3 of k basis vector
        :return: TLorentzVector in (n, r, k) basis
        """
        lepton_vec = lepton.Vect()
        return TLorentzVector(
            lepton_vec.Dot(n),
            lepton_vec.Dot(r),
            lepton_vec.Dot(k),
            lepton.E()
        )

    def w_rest_booster(self, part):
        """
        Boosts particles to W rest frames and maps leptons to (n, r, k) basis.
        :param part: np.array of shape (16,) representing [WpBoson(4), WpLepton(4), WnBoson(4), WnLepton(4)]
        :return: tuple of np.arrays (w_rest_WpLepton, w_rest_WnLepton)
        """
        # unpack input 4-vectors
        WpBoson = TLorentzVector(*part[:4])
        WpLepton = TLorentzVector(*part[4:8])
        WnBoson = TLorentzVector(*part[8:12])
        WnLepton = TLorentzVector(*part[12:16])
        
        # Step 1: Construct Higgs 4-vector and boost all particles to Higgs rest frame
        Higgs = WpBoson + WnBoson
        Beam_p = TLorentzVector(0, 0, 1, 1) # spatial-axis
        Higgs_boost = Higgs.BoostVector()
        self._boost_to_rest_frame([WpBoson, WpLepton, WnBoson, WnLepton, Beam_p], Higgs_boost)

        # Step 2: Construct orthogonal basis (k, r, n)
        k, r, n = self._construct_basis(WpBoson, Beam_p)

        # Step 3: Boost to W+ and W- rest frames
        self._boost_to_rest_frame([WpBoson, WpLepton], WpBoson.BoostVector())
        self._boost_to_rest_frame([WnBoson, WnLepton], WnBoson.BoostVector())

        # Step 4: Map leptons to (n, r, k) basis
        WpLp_k = self._map_to_basis(WpLepton, n, r, k)
        WnLp_k = self._map_to_basis(WnLepton, n, r, k)

        # lepton 4-vector in their parent W rest frames  
        w_rest_WpLepton = np.array([WpLp_k.X(), WpLp_k.Y(), WpLp_k.Z(), WpLp_k.T()]) # np.array([px, py, pz, energy])
        w_rest_WnLepton = np.array([WnLp_k.X(), WnLp_k.Y(), WnLp_k.Z(), WnLp_k.T()])

        return w_rest_WpLepton, w_rest_WnLepton

    def setup(self, workers=8):
        """
        Sets up the booster by processing all particles.
        (!) You need to call this method before accessing boosted leptons related attributes.
        :param workers: Number of parallel workers to use (default=8). If <=0, uses all available CPU cores.
        """

        if workers <= 0 or workers > os.cpu_count():
            # only use ~25% of available CPU cores
            workers = os.cpu_count() // 4
            print(f"Setting up booster with {workers} workers...")
        with multiprocessing.Pool(workers) as pool:
			# Retrieve the output from the pool
            results = list(pool.map(self.w_rest_booster, self.particles))
        w_rest_lp, w_rest_ln = zip(*results)
        self.w_rest_lp, self.w_rest_ln = np.vstack(w_rest_lp), np.vstack(w_rest_ln)
        print("Booster setup completed.")
        
    def lep_4_in_w_rest(self):
        """
        Returns the lepton 4-vectors in W rest frames.
        :return: tuple of np.arrays (w_rest_WpLepton, w_rest_WnLepton)
        """
        try:
            return self.w_rest_lp, self.w_rest_ln
        except AttributeError:
            print("Booster not set up yet. Please call setup() first.")
            return None, None

    def lep_theta_phi_in_w_rest(self):
        """
        Returns the lepton theta and phi angles (normalized) in their parent W rest frames.
        :return: tuple of ((pos_lep_theta, pos_lep_phi), (neg_lep_theta, neg_lep_phi))
        """
        try:
            _ = self.w_rest_lp
            _ = self.w_rest_ln
        except AttributeError:
            print("Booster not set up yet. Please call setup() first.")
            return None, None

        def theta(p4):
            p3_mag = np.sqrt(np.sum(np.square(p4[:, 0:3]), axis=1))
            pz = p4[:, 2]
            return np.arccos(np.divide(pz, p3_mag)) / np.pi  # Normalize to [0, 1]

        def phi(p4):
            phi = np.arctan2(p4[:, 1], p4[:, 0])
            return phi / np.pi  # normalize to [-1, 1]

        pos_lep_theta = theta(self.w_rest_lp)
        pos_lep_phi = phi(self.w_rest_lp)
        neg_lep_theta = theta(self.w_rest_ln)
        neg_lep_phi = phi(self.w_rest_ln)

        return (pos_lep_theta, pos_lep_phi), (neg_lep_theta, neg_lep_phi)
    
    def lep_xi_in_w_rest(self):
        """
        Returns the lepton directional cosines of leptons in their parent W rest frames.
        :return: tuple of ((pos_lep_xi_n, pos_lep_xi_r, pos_lep_xi_k), (neg_lep_xi_n, neg_lep_xi_r, neg_lep_xi_k))
        """
        try:
            _ = self.w_rest_lp
            _ = self.w_rest_ln
        except AttributeError:
            print("Booster not set up yet. Please call setup() first.")
            return None, None
        
        def xi(p4):
            xi_n = p4[:, 0] / np.linalg.norm(p4[:, :3], axis=1)
            xi_r = p4[:, 1] / np.linalg.norm(p4[:, :3], axis=1)
            xi_k = p4[:, 2] / np.linalg.norm(p4[:, :3], axis=1)
            return xi_n, xi_r, xi_k

        xi_pos = xi(self.w_rest_lp)
        xi_neg = xi(self.w_rest_ln)

        return xi_pos, xi_neg
    
    def _cglmp(self, z_xp, z_xn, z_yp, z_yn):
        """
        Calculate Bxy (CGLMP values).
        (x, y) -> (n, r) basis; It can generlaize to (y, z) and (z, x) basis as well.
        :param z_xp: directional cosine (xi) onto x basis for positive lepton
        :param z_xn: directional cosine (xi) onto x basis for negative lepton
        :param z_yp: directional cosine (xi) onto y basis for positive lepton
        :param z_yn: directional cosine (xi) onto y basis for negative lepton
        :return: Bxy value
        """
        tr_a = (np.divide(8, np.sqrt(3))) * (z_xp * z_xn + z_yp * z_yn)
        tr_b = 25 * (np.square(z_xp) - np.square(z_yp)) * (np.square(z_xn) - np.square(z_yn))
        tr_c = 100 * (z_xp * z_yp * z_xn * z_yn)
        return tr_a + tr_b + tr_c
    
    def cglmp_bij(self):
        """
        Calculate CGLMP Bxy, Byz, Bzx values.
        """
        try:
            _ = self.w_rest_lp
            _ = self.w_rest_ln
        except AttributeError:
            print("Booster not set up yet. Please call setup() first.")
            return None
        
        xi_pos, xi_neg = self.lep_xi_in_w_rest()
        b_xy = self._cglmp(xi_pos[0], xi_neg[0], xi_pos[1], xi_neg[1])
        b_yz = self._cglmp(xi_pos[1], xi_neg[1], xi_pos[2], xi_neg[2])
        b_zx = self._cglmp(xi_pos[0], xi_neg[0], xi_pos[2], xi_neg[2])
        return b_xy, b_yz, b_zx

if __name__ == "__main__":
    import time
    t1 = time.time()
    
    # Example usage
    # load sample data
    import load_data
    data = load_data.load_particles_from_h5("/root/data/mc20_truth_v4_SM.h5")
    # pack 4-vectors of leptons and  W bosons
    particles = np.concatenate(
		[
			data["pos_w"]["p4"],
			data["truth_pos_lep"]["p4"],
			data["neg_w"]["p4"],
			data["truth_neg_lep"]["p4"],
		],
		axis=-1,
	)
    print(particles.shape)
    booster = Booster(particles)
    booster.setup() # after construction, you need to call setup() method first
    # Pick up the boosted leptons related attributes you wanna use
    # 1. Get leptons theta and phi in W rest frames
    lep_pos_ang, lep_neg_ang = booster.lep_theta_phi_in_w_rest()
    lep_pos_theta, lep_pos_phi = lep_pos_ang
    lep_neg_theta, lep_neg_phi = lep_neg_ang
    # 2. Get leptons 4-vectors in W rest frames
    lep_pos_p4, lep_neg_p4 = booster.lep_4_in_w_rest()
    # 3. Get leptons directional cosines in W rest frames
    lep_pos_xi, lep_neg_xi = booster.lep_xi_in_w_rest()
    lep_pos_xi_n, lep_pos_xi_r, lep_pos_xi_k = lep_pos_xi
    lep_neg_xi_n, lep_neg_xi_r, lep_neg_xi_k = lep_neg_xi
    # 4. Calculate CGLMP Bxy, Byz, Bzx values
    b_xy, b_yz, b_zx = booster.cglmp_bij()
    
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1:<.2f} seconds")