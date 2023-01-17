import csv
import pandas as pd
import os

def create_dir(dir):
  """ function for creating empty folders
  """
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir

def cvs_txt(filename, cluster_list, folder_list):
  """ function for putting excel documents into folders
  """
  # open CVS file
  df = pd.read_csv(filename)
  df = df.fillna('')

  # print(df.shape)

  ## put the rows into corresponding clustered folders
  # loop clusterName in the cluster_list
  for clusterName in cluster_list:

    posit = cluster_list.index(clusterName)

    # produce dataframe for clustered rows
    classified_df = df[df.Classification == clusterName]
    classified_df = classified_df.reset_index()
    classified_df = classified_df[['article', 'abstract', 'keywords']]
    classified_df['text'] = classified_df['article'] + '. ' + classified_df['abstract'] + ' ' + classified_df[
      'keywords']

    # create sequential numbers for naming files
    textNames = [i for i in range(classified_df.shape[0])]

    print(classified_df.shape)

    # put each row into corresponding files
    for textName in textNames:
      f = open("data/" + folder_list[posit] + "/" + "scr" + str(textName)
               + ".txt", 'w', encoding="utf-8")  # the notation can be adapteed based on your project
      f.write(classified_df['text'][textName])
      f.close()




# name all folders (names can be changed based on your own project)
folder_list = ['1_network_design', '2_decision_system', '3_supply_mgmt', '4_chain_coordination',
               '5_behavior_chain', '6_buiding_SCR', '7_framework_design', '8_theory', '9_risk_mitigation',
               '10_implication', '11_factors']

# # creat folders (uncomment to create empty folders to contain classified files)
# for name in folder_list:
#   path = "E:/pythonProject/scrCovid/data/" + name
#   create_dir(path)

# present the link for the CSV file that you are going to read
filename = 'data/sample.csv'

# create txt files for each cluster folder
cluster_list = ['Network Design', 'Decision support and measurement system', 'Supply and inventory management',
                'Supply chain coordination', 'Behavior supply chain', 'Approaches for building SCR', 'General framework design',
                'Theoretical Underpinnings', 'Risk Mitigation', 'Operational and Financial implication of SCR/SCD',
                'Factors that affect SCR']

# scrap text in cvs file to .txt file and put them into corresponding folders
cvs_txt(filename, cluster_list, folder_list)