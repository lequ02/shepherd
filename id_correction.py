import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import os



prev_frame_file = "result\\track6 - Copy\\labels_xyxy\\output_video4_1.txt"
next_frame_file = "result\\track6 - Copy\\labels_xyxy\\output_video4_500.txt"

title = ["label", "xtop", "ytop", "xbot", "ybot", "id"]
df1 = pd.read_csv(prev_frame_file, header = None, names=title, sep=" ")
df2 = pd.read_csv(next_frame_file, header = None, names=title, sep=" ")
# df1.set_index("id", inplace=True)
# df2.set_index("id", inplace=True)


id_mapping_cos_sim = {}
id_mapping_manhattan = {}

# Rows in df2 with ID>32:
lost_track2 = df2[df2.id>32]

# Rows with ID that is in df1 but not in df2
# Create a boolean mask indicating whether each ID in df1 appears in df2
ids_in_df2 = df1['id'].isin(df2['id'])
# Filter df1 based on the boolean mask
lost_track1 = df1[~ids_in_df2]
print("lost_track1\n", lost_track1)
# print(lost_track1)
print("\nlost_track2\n",lost_track2)

cos_sim = pd.DataFrame(cosine_similarity(lost_track1.iloc[:,1:-1], lost_track2.iloc[:,1:-1]))

distance_matrix = pairwise_distances(lost_track1.iloc[:,1:-1], lost_track2.iloc[:,1:-1], metric='manhattan')

def map_id_cos_sim(cos_sim_matrix):

    while not np.all(np.isinf(cos_sim_matrix)):
        # print("\n",i)
        # print(cos_sim)
        # print("max cos sim" ,np.argmax(cos_sim))
        max_index = np.argmax(cos_sim_matrix)
        max_row, max_col = np.unravel_index(max_index, cos_sim_matrix.shape)
        id_mapping_cos_sim[lost_track2.id.iloc[max_col]] = lost_track1.id.iloc[max_row]
        print("cos sim",lost_track1.id.iloc[max_row], lost_track2.id.iloc[max_col])
        cos_sim_matrix.iloc[max_row, :] = -np.inf
        cos_sim_matrix.iloc[:, max_col] = -np.inf

def map_id_manhattan(distance_matrix):
    while not np.all(np.isinf(distance_matrix)):
        # print(distance_matrix)
        # print("min manhattan", np.argmin(distance_matrix))
        min_index = np.argmin(distance_matrix)
        min_row, min_col = np.unravel_index(min_index, distance_matrix.shape)
        id_mapping_manhattan[lost_track2.id.iloc[min_col]] = lost_track1.id.iloc[min_row]
        print("manhattan",lost_track1.id.iloc[min_row], lost_track2.id.iloc[min_col])
        distance_matrix[min_row, :] = np.inf
        distance_matrix[:, min_col] = np.inf

map_id_cos_sim(cos_sim)
map_id_manhattan(distance_matrix)

# Create new dataframes for cosine similarity and Manhattan distance mappings
df_cos_sim = df2.copy()
df_manhattan = df2.copy()

# Replace IDs in the new dataframes with cosine similarity and Manhattan distance mappings
df_cos_sim.loc[df_cos_sim['id'].isin(lost_track2['id']), 'id'] = df_cos_sim['id'].map(id_mapping_cos_sim)
df_manhattan.loc[df_manhattan['id'].isin(lost_track2['id']), 'id'] = df_manhattan['id'].map(id_mapping_manhattan)

# Write to files
output_file_cos_sim = "result/track6 - Copy/reID/output_video4_500_cos_sim_reID.txt"
output_file_manhattan = "result/track6 - Copy/reID/output_video4_500_manhattan_reID.txt"

with open(output_file_cos_sim, 'w') as f:
    f.write(df_cos_sim.to_csv(index=False, sep=" ", header=False))

with open(output_file_manhattan, 'w') as f:
    f.write(df_manhattan.to_csv(index=False, sep=" ", header=False))

print("Cosine Similarity ID Mapping:")
print(id_mapping_cos_sim)
print("\nManhattan Distance ID Mapping:")
print(id_mapping_manhattan)


# print(df1)
# print(df2)