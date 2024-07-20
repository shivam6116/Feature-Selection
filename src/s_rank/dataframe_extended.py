########################################################################
#           DATAFRAME EXTENDED                                         #
########################################################################
#the structure is an extension of the pandas.Dataframe structure.
#it contains additional information such as the list of numerical variables, 
#non-numerical variables, the matrix of distances and similarities.
#also, given a dataset on construction, ot performs rescaling, discretization and
#removal of outliers
import numpy as np
import pandas as pd 
import math
import time
import sys
from sklearn import preprocessing
from scipy import stats
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
class dataframe_ext:
    #########################################
    # Attributes: 
    #   - dataFrame:                df
    #   - type of varaibles:        TYPE                     
    #   - discrete variables        discrete_vars -> None if TYPE = continous
    #                                                all if TYPE = discrete
    #                                                list provided if TYPE = mixed
    #   - continous variables       continous_vars -> same logic as discrete
    #   - object variables          object_vars -> variables of type object
    #   - distance type:            dist          -> euclidean if TYPE = continous
    #                                                hamming otherwise
    #   - distances matrix:         D
    #   - similarities matrix:      S
    #   - similarity parameter:     alpha
    #   - entropy value:            E
    #########################################

    #########################################
    # CONSTRUCTOR - most of the attributes  #
    # are actually blank until complete     #
    # is called                             #
    #########################################
    def __init__(self, dataframe, vars_type, discrete_vars_list = None, clean_bool = False, rescale_bool = False):
        #the data is stored
        self.dist_type = None
        if type(dataframe) == pd.core.frame.DataFrame:
            self.df = dataframe.dropna()
        else: 
            sys.exit("Please specify a valid dataframe")

        #first we set the list of object_vars
        self.object_vars = [x for x in self.df.columns if self.df[x].dtype == "object"]
        #then a little control over inferred types
        if len(self.object_vars) != 0 and vars_type == "continous":
            sys.exit("vars_type is set to continous but object type columns are present.\
                      If there are non numerical variables in the dataset please set \
                      vars_type = mixed, otherwise make sure that dtypes in the \
                      dataframe are inferred correctly.")

        #according to vars_type, every parameter is set
        if vars_type not in ["continous", "mixed", "discrete"]:
            sys.exit("Please specify a valid vars_type: [continous, discrete, mixed]")

        elif vars_type == "continous":
            self.TYPE = vars_type
            self.dist = self.euclidean_distance
            self.dist_type = "euclidean"
            self.discrete_vars = []
            self.continous_vars = self.df.drop(columns = self.object_vars)
            self.clean_and_rescale(clean_bool, rescale_bool)

        elif vars_type == "discrete":
            self.TYPE = vars_type
            self.dist = self.hamming_distance
            self.dist_type = "hamming"
            #self.discrete_vars = discrete_vars_list
            self.discrete_vars = self.df.columns
            self.continous_vars = []
            self.clean_and_rescale(clean_bool, rescale_bool)


        elif vars_type == "mixed":
            self.TYPE = vars_type
            self.dist = self.hamming_distance
            self.dist_type= "hamming"
            print(type(discrete_vars_list))
            if type(discrete_vars_list) != list:
                sys.exit("Please provide a valid discrete variable list")
            else:
                try: 
                    _ = self.df[discrete_vars_list]
                except:
                    sys.exit("Columns in discrete_var_list are not in the Dataframe")
                self.discrete_vars = discrete_vars_list
                self.continous_vars = (
                                        self.df
                                            .drop(columns = self.discrete_vars + self.object_vars)
                                            .columns
                                      )
                self.clean_and_rescale(clean_bool, rescale_bool)
                self.discretize()
        
        #all the other parameters are set to None
        self.D = None
        self.S = None
        self.alpha = None
        self.E = None

    #########################################
    # COMPLETE - determines the value of    #
    # all the other attributes in the right #
    # order                                 #
    #########################################
    def complete(self):
        '''original version'''                   
        self.dist_matrix()
        self.set_alpha()
        self.sim_matrix()
        self.simmatrix_entropy()



    def complete1(self):
        '''new version'''                    
        start_time = time.time()
        # print(len(self.columns))
        self.dist_matrix()
        dist_matrix_time = time.time() - start_time
        print(f"Time taken by dist_matrix: {dist_matrix_time} seconds")

        start_time = time.time()
        self.set_alpha()
        set_alpha_time = time.time() - start_time
        print(f"Time taken by set_alpha: {set_alpha_time} seconds")

        start_time = time.time()
        self.sim_matrix()
        sim_matrix_time = time.time() - start_time
        print(f"Time taken by sim_matrix: {sim_matrix_time} seconds")

        start_time = time.time()
        self.simmatrix_entropy()
        simmatrix_entropy_time = time.time() - start_time
        print(f"Time taken by simmatrix_entropy: {simmatrix_entropy_time} seconds")


    #########################################
    # CLEAN - removes all rows that have    #
    # outliers in them, i.e. values outside #
    # the interval mean +- 3*devstd         #
    #########################################
    def clean(self):
        object_df = self.df[self.object_vars]
        numeric_df = self.df.drop(columns = self.object_vars)
        ZS = stats.zscore(numeric_df)
        numeric_df = numeric_df[(np.abs(ZS) < 3).all(axis = 1)]
        self.df = numeric_df.join(object_df)
        return self
    
    #########################################
    # RESCALE - Rescale variables to have   #
    # values from 0 to 1                    #
    #                                       #
    #########################################
    def rescale(self):
        scaler = preprocessing.MinMaxScaler()
        object_df = self.df[self.object_vars]
        numeric_df = self.df.drop(columns = self.object_vars)
        numeric_df = pd.DataFrame(
                                  scaler.fit_transform(numeric_df),
                                  columns = numeric_df.columns
                                 )
        self.df = numeric_df.join(object_df)
        return self
    
    #########################################
    # CLEAN AND RESCALE - small utility     #
    # function that performs clean and      #
    # rescale if the parameters are True    #
    #########################################
    def clean_and_rescale(self, clean_bool = True, rescale_bool = True):
        if clean_bool == True:
            self.clean()
        if rescale_bool == True:
            self.rescale()
        return self
    
    #########################################
    # DISCRETIZE - makes all the variables  #
    # discrete (could have guessed that)    #
    #                                       #
    #########################################
    def discretize(self):
        disc = preprocessing.KBinsDiscretizer(n_bins = 10,                   # Watch here: you might have to tweak this parameter a bit
                                              encode = "ordinal",            # I'm currently looking for a way to make this process automatic
                                              strategy = "uniform")          # If the data is standardized and rescaled though, 10 might be a good 
        continous_df = self.df[self.continous_vars]                          # first attempt
        not_continous_df = self.df.drop(columns = self.continous_vars)
        continous_df = pd.DataFrame(
                                        disc.fit_transform(continous_df), 
                                        columns = continous_df.columns
                                   )
        self.df = continous_df.join(not_continous_df)
        return self
    

    #########################################
    # EUCLIDEAN DISTANCE - the distance     #
    # used if the data contains only        #
    # numerical attriubtes                  #
    #########################################
    

    @staticmethod 
    def euclidean_distance(x1, x2):
        '''previous version'''
        # return np.linalg.norm(x1 - x2)
        # print("euclidean distance")
        '''new version'''

        return distance.euclidean(x1 , x2)
    

    #########################################
    # HAMMING DISTANCE - the distance       #
    # used if the data contains at least    #
    # one discrete attribute                #
    #########################################
    @staticmethod
    def hamming_distance(x1, x2):
        """
        A function to calculate the Hamming distance between two sequences x1 and x2.
        """
        # print("------Hamming------")

        dist = 0
        for x1i, x2i in zip(x1, x2):
            if x1i != x2i:
                dist = dist + 1
        return dist / len(x1)

    def dist_matrix(self):
        '''
            this is to handle the fact that the dataframe
            could be a sample of a larger one
        '''
        df = self.df.reset_index(drop = True)
        if self.dist_type == "euclidean":
            pairwise_distances = pdist(df.values, metric='euclid')
            self.D = np.matrix(squareform(pairwise_distances))
        else:
             self.D = np.asmatrix(
                    [
                        [
                            self.dist(row1, row2) 
                            for _, row1 in df.iterrows()
                        ]
                        for _, row2 in df.iterrows()
                    ]
                )


    
    #########################################
    # SET_ALPHA - sets the parameter for    #
    # the computation of similarity         #
    # requires D to be set                  #
    #########################################
    def set_alpha(self):
        self.alpha = -math.log(0.5) / np.matrix.mean(self.D)

    #########################################
    # SIM - converts a distance measure in  #
    # a similarity one                      #
    # requires alpha                        #
    #########################################
    def sim(self, dist_value):
        return math.exp(- self.alpha * dist_value)

    #########################################
    # SIM_MATRIX - calculates the           #
    # similarity matrix                     #
    #                                       #
    #########################################
    def sim_matrix(self):
        #vectorizing sim  
        vsim = np.vectorize(self.sim)
        #setting the similarity matrix
        self.S = vsim(self.D)
    
    #########################################
    # SIM_TO_ENTROPY - convert a similarity #
    # value into an entropy                 #
    #                                       #
    #########################################
    @staticmethod
    def sim_to_entropy(sv):
        try:
            return - ((sv * math.log2(sv)) + ((1-sv)*math.log2(1-sv)))
        except: 
            return float(0)
            
    
    #########################################
    # SIMMATRIX_ENTROPY - calculates the    #
    # entropy of the dataset                #
    #                                       #
    #########################################
    def simmatrix_entropy(self):
        entropies = [self.sim_to_entropy(x) for x in np.nditer(self.S)]
        self.E = np.nansum(entropies)/2 #FLOAT
