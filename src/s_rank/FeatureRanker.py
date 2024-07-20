import pandas as pd 
import numpy as np 
import math
from src.s_rank.dataframe_extended import dataframe_ext
import warnings
import concurrent.futures
warnings.filterwarnings("ignore")


class SRANK:
    ###########################################################################
    # Attributes: 
    # - info : a dataframe that contains all the variables as rows, with 
    #          the score obtained 
    # - rank : a list that contains the feature names in order of importance
    ###########################################################################
    def __init__(self):
        self.info = None
        self.rank = None 

    ###########################################################################
    # THIS ALGORITHM IS USED INSIDE SRANK.apply - it performs the basic       #
    # operations - more info on the paper cited                               #
    ###########################################################################
    def RANK(self, df, vars_type):
        #list of the variables
        features = list(df.columns)

        #empty dictionary that will contain the entropy 
        #values for each variable
        entropy_values = []
        
        #now for each variable
        print("\t", "features: ")
        for F in features:
            print("\t\t", F)
            #create a new dataframe_ext, without the
            #variable F
            feature_partial = dataframe_ext(dataframe = df.drop(columns = [F]), 
                                            vars_type = vars_type
                                           )
            #calculate all the quantities
            feature_partial.complete()
            #retrieving the Entropy for feature_partial
            entropy_values.append(feature_partial.E)
        
        #creating a dataframe from the entropy values obtained
        rankings = pd.DataFrame({
                                    "feature" : features, 
                                    "entropy" : entropy_values
                                })

        #making it suitable
        rankings = (rankings.sort_values(by = "entropy")                 #sort according to entropy values
                        .reset_index(drop = True)                        #reset index
                        .reset_index()                                   #reset again to have the ranking columns
                        .rename(columns = {"index" : "score"})           #rename index -> score
                        .set_index("feature"))                           #putting index on feature name
        return rankings


    ###########################################################################
    # S-RANK algorithm, every information is contained in the paper cited     #
    # many operations are taken care of inside dataframe_ext                  #
    ###########################################################################

    def process_sample(self, df_big, SAMPLE_SIZE, vars_type, i):
        print("N_SAMPLE: ", i)
        # random sampling of df_big
        df_sample = df_big.sample(n=SAMPLE_SIZE, replace=True)
        sample_ranking = self.RANK(df_sample, vars_type)
        return sample_ranking["score"]

    def apply(self, df_big, vars_type, discrete_var_list, 
              clean_bool = False, rescale_bool = None, 
              N_SAMPLE = 40, SAMPLE_SIZE = 50):


        #initializing the ranking dataframe, initially everything is set to 0
        features = list(df_big.columns)
        print("New")
        orankings = [0 for x in features]
        tot_rankings = pd.DataFrame({
                                        "feature" : features, 
                                        "score_final" : orankings,
                                    }).set_index("feature")
        
        #defining the first complete dataframe, and performing rescaling and cleaning
        #if necessary
        df_big = dataframe_ext(dataframe = df_big, 
                               vars_type = vars_type, 
                               discrete_vars_list = discrete_var_list, 
                               clean_bool = clean_bool, 
                               rescale_bool = rescale_bool).df

        #now that the dataset has been discretized, we can change the var type
        if vars_type == "mixed":
            vars_type = "discrete"


        #now for each i, we take a sample and apply RANK
        ##for i in range(N_SAMPLE):
        ##   print("SAMPLE: ", i)
            #random sampling of df_big
        #df_sample = df_big.sample(n = SAMPLE_SIZE, replace = True)
        #sample_ranking = self.RANK(df_sample, vars_type)
        #now we sum the scores obtained with the previous 
        #tot_rankings["score_final"] = tot_rankings["score_final"] + sample_ranking["score"]
        #final_results_info = tot_rankings.sort_values(by = "score_final", ascending = False)
        #final_results_rank = final_results_info.reset_index()["feature"] 
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_sample, df_big, SAMPLE_SIZE, vars_type, i) for i in range(N_SAMPLE)]
        
        for future in concurrent.futures.as_completed(futures):
            sample_score = future.result()
            tot_rankings["score_final"] = tot_rankings["score_final"] + sample_score
        
        final_results_info = tot_rankings.sort_values(by="score_final", ascending=False)
        final_results_rank = final_results_info.reset_index()["feature"]
        
        #setting the class attributes
        self.info = final_results_info
        self.rank = final_results_rank
        
    
        ########################################################
        

        return self
