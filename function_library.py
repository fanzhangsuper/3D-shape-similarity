import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import sys
import math
import matplotlib.tri
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.isotonic import IsotonicRegression
import networkx as nx


class function_library:
    def triangle_csc(pts):
        rows, cols = pts.shape
    
        A = np.bmat([[2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                     [np.ones((1, rows)), np.zeros((1, 1))]])
    
        b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
        x = np.linalg.solve(A,b)
        bary_coords = x[:-1]
        return np.sum(pts * np.tile(bary_coords.reshape((pts.shape[0], 1)), (1, pts.shape[1])), axis=0)
        
    def voronoi(P):
        delauny = Delaunay(P)
        triangles = delauny.points[delauny.vertices]
    
        lines = []
    
        # Triangle vertices
        A = triangles[:, 0]
        B = triangles[:, 1]
        C = triangles[:, 2]
        lines.extend(zip(A, B))
        lines.extend(zip(B, C))
        lines.extend(zip(C, A))
        lines = matplotlib.collections.LineCollection(lines, color='r')
        plt.gca().add_collection(lines)
    
        circum_centers = np.array([function_library.triangle_csc(tri) for tri in triangles])
    
        segments = []
        for i, triangle in enumerate(triangles):
            circum_center = circum_centers[i]
            for j, neighbor in enumerate(delauny.neighbors[i]):
                if neighbor != -1:
                    segments.append((circum_center, circum_centers[neighbor]))
                else:
                    ps = triangle[(j+1)%3] - triangle[(j-1)%3]
                    ps = np.array((ps[1], -ps[0]))
    
                    middle = (triangle[(j+1)%3] + triangle[(j-1)%3]) * 0.5
                    di = middle - triangle[j]
    
                    ps /= np.linalg.norm(ps)
                    di /= np.linalg.norm(di)
    
                    if np.dot(di, ps) < 0.0:
                        ps *= -1000.0
                    else:
                        ps *= 1000.0
                    segments.append((circum_center, circum_center + ps))
        return segments
    
    def best_fit_transform(A, B):
        '''
        Calculates the least-squares best-fit transform between corresponding 3D points A->B
        Input:
          A: Nx3 numpy array of corresponding 3D points
          B: Nx3 numpy array of corresponding 3D points
        Returns:
          T: 4x4 homogeneous transformation matrix
          R: 3x3 rotation matrix
          t: 3x1 column vector
        '''
    
        assert len(A) == len(B)
    
        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
    
        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
    
        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = np.dot(Vt.T, U.T)
    
        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)
    
        # homogeneous transformation
        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
    
        return T, R, t
    
    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nx3 array of points
            dst: Nx3 array of points
        Output:
            distances: Euclidean distances (errors) of the nearest neighbor
            indecies: dst indecies of the nearest neighbor
        '''
    
        indecies = np.zeros(src.shape[0], dtype=np.int)
        distances = np.zeros(src.shape[0])
        for i, s in enumerate(src):
            min_dist = np.inf
            for j, d in enumerate(dst):
                dist = np.linalg.norm(s-d)
                if dist < min_dist:
                    min_dist = dist
                    indecies[i] = j
                    distances[i] = dist
        return distances, indecies
    
    def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-3):
        '''
        The Iterative Closest Point method
        Input:
            A: Nx3 numpy array of source 3D points
            B: Nx3 numpy array of destination 3D point
            init_pose: 4x4 homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation
            distances: Euclidean distances (errors) of the nearest neighbor
        '''
        N = len(A)
        # make points homogeneous, copy them so as to maintain the originals
        src = np.ones((3,A.shape[0]))
        dst = np.ones((3,B.shape[0]))
        src[0:3,:] = np.copy(A.T)
        dst[0:3,:] = np.copy(B.T)
    
        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)
    
        prev_error = 0
        # define a dic to store the source at each step
        dic = {}
        for i in range(max_iterations):
            print('at iteration:', i)
            dic[i] = src
            # find the nearest neighbours between the current source and destination points
            distances, indices = function_library.nearest_neighbor(src[0:3,:].T, dst[0:3,:].T)
            print('compute nearest neighbor finished')
            # compute the transformation between the current source and nearest destination points
            T, R, t = function_library.best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)
            print('get the best fit trasform at current iteration')
            # update the current source
            src = R.dot(src) + np.tile(t[:,None], (1, N))
    
            # check error
            mean_error = np.sum(distances) / distances.size
            if abs(prev_error-mean_error) < tolerance:
                break
            prev_error = mean_error
            
            print('the error is:', prev_error)
    
        # calculcate final tranformation
        T, R, t = function_library.best_fit_transform(A, src[0:3,:].T)
    
        return T, distances, dic

    def reading_text(path):
	    vertice = list()
	    with open(path) as f:
	        for line in f:
	            vertice.append([float(x) for x in line.split( )])
	    vertice = np.asarray(vertice)
	    return vertice

    # compute the (source, dest, weight) tuple, weight is represented by its 3D distance
    def convert_to_edge(polygon_edge, vertice):
	    # compute all the (source, dest) pair
	    b = np.insert(polygon_edge, 3, values=polygon_edge[:,0], axis=1)
	    b1 = b[:,0:2]; b2 = b[:,1:3]; b3 = b[:,2:]
	    all_pair = np.insert(b1, len(b1), values=b2, axis=0)
	    all_pair = np.insert(all_pair, len(all_pair), values=b3, axis=0)
	    # compute the distance between each pair
	    dis = np.zeros((len(all_pair),1))
	    for i in range(len(all_pair)):
	        dis[i,0] = np.linalg.norm(vertice[int(all_pair[i,0]-1),:] - vertice[int(all_pair[i,1]-1),:])
	        edge = np.insert(all_pair, 2, values=dis[:,0], axis=1)
	    return edge
    
     
    # Voronoi tessellation
    def voronoi2(P, bbox=None):
       
        def circumcircle2(T):
            P1,P2,P3=T[:,0], T[:,1], T[:,2]
            b = P2 - P1
            c = P3 - P1
            d=2*(b[:,0]*c[:,1]-b[:,1]*c[:,0])
            center_x=(c[:,1]*(np.square(b[:,0])+np.square(b[:,1]))- b[:,1]*
                      (np.square(c[:,0])+np.square(c[:,1])))/d + P1[:,0]
            center_y=(b[:,0]*(np.square(c[:,0])+np.square(c[:,1]))- c[:,0]*
                      (np.square(b[:,0])+np.square(b[:,1])))/d + P1[:,1]
            return np.array((center_x, center_y)).T
    
        def check_outside(point, bbox):
            point=np.round(point, 4)
            return point[0]<bbox[0] or point[0]>bbox[2] or point[1]< bbox[1] or point[1]>bbox[3]
    
        def move_point(start, end, bbox):
            vector=end-start
            c=calc_shift(start, vector, bbox)
            if c is not None:
                if (c>0 and c<1):
                    start=start+c*vector
                    return start
    
        def calc_shift(point, vector, bbox):
            c=sys.float_info.max
            for l,m in enumerate(bbox):
                a=(float(m)-point[l%2])/vector[l%2]
                if  a>0 and  not check_outside(point+a*vector, bbox):
                    if abs(a)<abs(c):
                        c=a
            return c if c<sys.float_info.max else None    
        
        if not isinstance(P, np.ndarray):
            P=np.array(P)
        if not bbox:
            xmin=P[:,0].min()
            xmax=P[:,0].max()
            ymin=P[:,1].min()
            ymax=P[:,1].max()
            xrange=(xmax-xmin) * 0.3333333
            yrange=(ymax-ymin) * 0.3333333
            bbox=(xmin-xrange, ymin-yrange, xmax+xrange, ymax+yrange)
        bbox=np.round(bbox,4)
     
        D = matplotlib.tri.Triangulation(P[:,0],P[:,1])
        T = D.triangles
        n = T.shape[0]
        C = circumcircle2(P[T])
     
        segments = []
        for i in range(n):
            for j in range(3):
                k = D.neighbors[i][j]
                if k != -1:
                    #cut segment to part in bbox
                    start,end=C[i], C[k]
                    if check_outside(start, bbox):
                        start=move_point(start,end, bbox)
                        if  start is None:
                            continue
                    if check_outside(end, bbox):
                        end=move_point(end,start, bbox)
                        if  end is None:
                            continue
                    segments.append( [start, end] )
                else:
                    #ignore center outside of bbox
                    if check_outside(C[i], bbox) :
                        continue
                    first, second, third=P[T[i,j]], P[T[i,(j+1)%3]], P[T[i,(j+2)%3]]
                    edge=np.array([first, second])
                    vector=np.array([[0,1], [-1,0]]).dot(edge[1]-edge[0])
                    line=lambda p: (p[0]-first[0])*(second[1]-first[1])/(second[0]-first[0])  -p[1] + first[1]
                    orientation=np.sign(line(third))*np.sign( line(first+vector))
                    if orientation>0:
                        vector=-orientation*vector
                    c=calc_shift(C[i], vector, bbox)
                    if c is not None:    
                        segments.append([C[i],C[i]+c*vector])
        return segments     
		
    """
    Multi-dimensional Scaling (MDS)
    """
    
    # author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
    # Licence: BSD

    def _smacof_single(similarities, metric=True, n_components=2, init=None,
                       max_iter=300, verbose=0, eps=1e-3, random_state=None):
        """
        Computes multidimensional scaling using SMACOF algorithm
    
        Parameters
        ----------
        similarities: symmetric ndarray, shape [n * n]
            similarities between the points
    
        metric: boolean, optional, default: True
            compute metric or nonmetric SMACOF algorithm
    
        n_components: int, optional, default: 2
            number of dimension in which to immerse the similarities
            overwritten if initial array is provided.
    
        init: {None or ndarray}, optional
            if None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array
    
        max_iter: int, optional, default: 300
            Maximum number of iterations of the SMACOF algorithm for a single run
    
        verbose: int, optional, default: 0
            level of verbosity
    
        eps: float, optional, default: 1e-6
            relative tolerance w.r.t stress to declare converge
    
        random_state: integer or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.
    
        Returns
        -------
        X: ndarray (n_samples, n_components), float
                   coordinates of the n_samples points in a n_components-space
    
        stress_: float
            The final value of the stress (sum of squared distance of the
            disparities and the distances for all constrained points)
    
        n_iter : int
            Number of iterations run.
    
        """
        similarities = check_symmetric(similarities, raise_exception=True)
    
        n_samples = similarities.shape[0]
        random_state = check_random_state(random_state)
    
        sim_flat = ((1 - np.tri(n_samples)) * similarities).ravel()
        sim_flat_w = sim_flat[sim_flat != 0]
        if init is None:
            # Randomly choose initial configuration
            X = random_state.rand(n_samples * n_components)
            X = X.reshape((n_samples, n_components))
        else:
            # overrides the parameter p
            n_components = init.shape[1]
            if n_samples != init.shape[0]:
                raise ValueError("init matrix should be of shape (%d, %d)" %
                                 (n_samples, n_components))
            X = init
    
        old_stress = None
        ir = IsotonicRegression()
        for it in range(max_iter):
            # Compute distance and monotonic regression
            dis = euclidean_distances(X)
    
            if metric:
                disparities = similarities
            else:
                dis_flat = dis.ravel()
                # similarities with 0 are considered as missing values
                dis_flat_w = dis_flat[sim_flat != 0]
    
                # Compute the disparities using a monotonic regression
                disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
                disparities = dis_flat.copy()
                disparities[sim_flat != 0] = disparities_flat
                disparities = disparities.reshape((n_samples, n_samples))
                disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                       (disparities ** 2).sum())
    
            # Compute stress
            stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
    
            # Update X using the Guttman transform
            dis[dis == 0] = 1e-5
            ratio = disparities / dis
            B = - ratio
            B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
            X = 1. / n_samples * np.dot(B, X)
    
            dis = np.sqrt((X ** 2).sum(axis=1)).sum()
            if verbose >= 2:
                print('it: %d, stress %s' % (it, stress))
            if old_stress is not None:
                if(old_stress - stress / dis) < eps:
                    if verbose:
                        print('breaking at iteration %d with stress %s' % (it,
                                                                           stress))
                    break
            old_stress = stress / dis
    
        return X, stress, it + 1
    
    
    def smacof(similarities, metric=True, n_components=2, init=None, n_init=8,
               n_jobs=1, max_iter=300, verbose=0, eps=1e-3, random_state=None,
               return_n_iter=False):
        """
        Computes multidimensional scaling using SMACOF (Scaling by Majorizing a
        Complicated Function) algorithm
    
        The SMACOF algorithm is a multidimensional scaling algorithm: it minimizes
        a objective function, the *stress*, using a majorization technique. The
        Stress Majorization, also known as the Guttman Transform, guarantees a
        monotone convergence of Stress, and is more powerful than traditional
        techniques such as gradient descent.
    
        The SMACOF algorithm for metric MDS can summarized by the following steps:
    
        1. Set an initial start configuration, randomly or not.
        2. Compute the stress
        3. Compute the Guttman Transform
        4. Iterate 2 and 3 until convergence.
    
        The nonmetric algorithm adds a monotonic regression steps before computing
        the stress.
    
        Parameters
        ----------
        similarities : symmetric ndarray, shape (n_samples, n_samples)
            similarities between the points
    
        metric : boolean, optional, default: True
            compute metric or nonmetric SMACOF algorithm
    
        n_components : int, optional, default: 2
            number of dimension in which to immerse the similarities
            overridden if initial array is provided.
    
        init : {None or ndarray of shape (n_samples, n_components)}, optional
            if None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array
    
        n_init : int, optional, default: 8
            Number of time the smacof algorithm will be run with different
            initialisation. The final results will be the best output of the
            n_init consecutive runs in terms of stress.
    
        n_jobs : int, optional, default: 1
    
            The number of jobs to use for the computation. This works by breaking
            down the pairwise matrix into n_jobs even slices and computing them in
            parallel.
    
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
            are used.
    
        max_iter : int, optional, default: 300
            Maximum number of iterations of the SMACOF algorithm for a single run
    
        verbose : int, optional, default: 0
            level of verbosity
    
        eps : float, optional, default: 1e-6
            relative tolerance w.r.t stress to declare converge
    
        random_state : integer or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.
    
        return_n_iter : bool
            Whether or not to return the number of iterations.
    
        Returns
        -------
        X : ndarray (n_samples,n_components)
            Coordinates of the n_samples points in a n_components-space
    
        stress : float
            The final value of the stress (sum of squared distance of the
            disparities and the distances for all constrained points)
    
        n_iter : int
            The number of iterations corresponding to the best stress.
            Returned only if `return_n_iter` is set to True.
    
        Notes
        -----
        "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
        Groenen P. Springer Series in Statistics (1997)
    
        "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
        Psychometrika, 29 (1964)
    
        "Multidimensional scaling by optimizing goodness of fit to a nonmetric
        hypothesis" Kruskal, J. Psychometrika, 29, (1964)
        """
    
        similarities = check_array(similarities)
        random_state = check_random_state(random_state)
    
        if hasattr(init, '__array__'):
            init = np.asarray(init).copy()
            if not n_init == 1:
                warnings.warn(
                    'Explicit initial positions passed: '
                    'performing only one init of the MDS instead of %d'
                    % n_init)
                n_init = 1
    
        best_pos, best_stress = None, None
    
        if n_jobs == 1:
            for it in range(n_init):
                pos, stress, n_iter_ = function_library._smacof_single(
                    similarities, metric=metric,
                    n_components=n_components, init=init,
                    max_iter=max_iter, verbose=verbose,
                    eps=eps, random_state=random_state)
                if best_stress is None or stress < best_stress:
                    best_stress = stress
                    best_pos = pos.copy()
                    best_iter = n_iter_
        else:
            seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
            results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
                delayed(function_library._smacof_single)(
                    similarities, metric=metric, n_components=n_components,
                    init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                    random_state=seed)
                for seed in seeds)
            positions, stress, n_iters = zip(*results)
            best = np.argmin(stress)
            best_stress = stress[best]
            best_pos = positions[best]
            best_iter = n_iters[best]
    
        if return_n_iter:
            return best_pos, best_stress, best_iter
        else:
            return best_pos, best_stress
    
    
    class MDS(BaseEstimator):
        """Multidimensional scaling
    
        Parameters
        ----------
        metric : boolean, optional, default: True
            compute metric or nonmetric SMACOF (Scaling by Majorizing a
            Complicated Function) algorithm
    
        n_components : int, optional, default: 2
            number of dimension in which to immerse the similarities
            overridden if initial array is provided.
    
        n_init : int, optional, default: 4
            Number of time the smacof algorithm will be run with different
            initialisation. The final results will be the best output of the
            n_init consecutive runs in terms of stress.
    
        max_iter : int, optional, default: 300
            Maximum number of iterations of the SMACOF algorithm for a single run
    
        verbose : int, optional, default: 0
            level of verbosity
    
        eps : float, optional, default: 1e-6
            relative tolerance w.r.t stress to declare converge
    
        n_jobs : int, optional, default: 1
            The number of jobs to use for the computation. This works by breaking
            down the pairwise matrix into n_jobs even slices and computing them in
            parallel.
    
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
            are used.
    
        random_state : integer or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.
    
        dissimilarity : string
            Which dissimilarity measure to use.
            Supported are 'euclidean' and 'precomputed'.
    
    
        Attributes
        ----------
        embedding_ : array-like, shape [n_components, n_samples]
            Stores the position of the dataset in the embedding space
    
        stress_ : float
            The final value of the stress (sum of squared distance of the
            disparities and the distances for all constrained points)
    
    
        References
        ----------
        "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
        Groenen P. Springer Series in Statistics (1997)
    
        "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
        Psychometrika, 29 (1964)
    
        "Multidimensional scaling by optimizing goodness of fit to a nonmetric
        hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    
        """
        def __init__(self, n_components=2, metric=True, n_init=4,
                     max_iter=300, verbose=0, eps=1e-3, n_jobs=1,
                     random_state=None, dissimilarity="euclidean"):
            self.n_components = n_components
            self.dissimilarity = dissimilarity
            self.metric = metric
            self.n_init = n_init
            self.max_iter = max_iter
            self.eps = eps
            self.verbose = verbose
            self.n_jobs = n_jobs
            self.random_state = random_state
    
        @property
        def _pairwise(self):
            return self.kernel == "precomputed"
    
        def fit(self, X, y=None, init=None):
            """
            Computes the position of the points in the embedding space
    
            Parameters
            ----------
            X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                    if dissimilarity='precomputed'
                Input data.
    
            init : {None or ndarray, shape (n_samples,)}, optional
                If None, randomly chooses the initial configuration
                if ndarray, initialize the SMACOF algorithm with this array.
            """
            self.fit_transform(X, init=init)
            return self
    
        def fit_transform(self, X, y=None, init=None):
            """
            Fit the data from X, and returns the embedded coordinates
    
            Parameters
            ----------
            X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                    if dissimilarity='precomputed'
                Input data.
    
            init : {None or ndarray, shape (n_samples,)}, optional
                If None, randomly chooses the initial configuration
                if ndarray, initialize the SMACOF algorithm with this array.
    
            """
            X = check_array(X)
            if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
                warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                              " dissimilarity matrix from data. To use a custom "
                              "dissimilarity matrix, set "
                              "``dissimilarity=precomputed``.")
    
            if self.dissimilarity == "precomputed":
                self.dissimilarity_matrix_ = X
            elif self.dissimilarity == "euclidean":
                self.dissimilarity_matrix_ = euclidean_distances(X)
            else:
                raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                                 " Got %s instead" % str(self.dissimilarity))
    
            self.embedding_, self.stress_, self.n_iter_ = function_library.smacof(
                self.dissimilarity_matrix_, metric=self.metric,
                n_components=self.n_components, init=init, n_init=self.n_init,
                n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
                eps=self.eps, random_state=self.random_state,
                return_n_iter=True)
    
            return self.embedding_
            
    def bbox(array, point, radius):
        a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
        b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
        c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
        return c
        
    def hausdorff(surface_a, surface_b):
    
        # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
        dists = []
    
        l = len(surface_a)
    
        for i in range(l):
    
            # walking through all the points of surface_a
            dist_min = 1000.0
            radius = 0
            b_mod = np.empty(shape=(0, 0, 0))
    
            # increasing the cube size around the point until the cube contains at least 1 point
            while b_mod.shape[0] == 0:
                b_mod = function_library.bbox(surface_b, surface_a[i], radius)
                radius += 1
    
            # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
            # increasing the size of the cube
            b_mod = function_library.bbox(surface_b, surface_a[i], radius * math.sqrt(3))
    
            for j in range(len(b_mod)):
                # walking through the small number of points to find the minimum distance
                dist = np.linalg.norm(surface_a[i] - b_mod[j])
                if dist_min > dist:
                    dist_min = dist
    
            dists.append(dist_min)
    
        return np.max(dists)

    def compute_distance(root_path, vertice_name, polygon_name, compute_distance=False):
        vertice = function_library.reading_text(root_path+vertice_name)
        polygon_edge = function_library.reading_text(root_path+polygon_name)
        print('Vertice and polygon are loaded!')
        # compute edge
        edge = function_library.convert_to_edge(polygon_edge, vertice)
        if compute_distance:
            G=nx.Graph()
            G.add_nodes_from(list(np.arange(1, len(vertice))))
            for i in range(len(edge)):
                G.add_edge(int(edge[i,0]),int(edge[i,1]),weight = edge[i,2])
            path = nx.all_pairs_dijkstra_path(G)
            print('Shortest path computed!')
            # compute the distance
            distance_matrix = np.zeros((len(vertice), len(vertice)))
            for iter1 in range(1, len(vertice)+1):
                for iter2 in range(1, len(vertice)+1):
                    if iter2>iter1:
                        node_path = path[iter1][iter2]
                        dis = 0
                        for i in range(len(node_path)-1):
                            dis = dis + np.linalg.norm(vertice[int(node_path[i]-1),:] 
                                                       - vertice[int(node_path[i+1]-1),:])    
                        distance_matrix[iter1-1,iter2-1] = dis
            distance_matrix = distance_matrix + distance_matrix.T
            print('Distance computed!')
            return distance_matrix
        else:
            return vertice, edge