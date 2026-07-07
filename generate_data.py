import pandas as pd
import numpy as np

np.random.seed(42)


# now we are creating library to simulate normal data in a connection
#  simulates 1000 normal connections with duration, packet size, number of connections and label 
normal_data = {
    "duration" : np.random.normal(50,10,1000),# Average 50 seconds with a standard deviation of 10 seconds
    "packet_size" : np.random.normal(500,50,1000),# Average 500 bytes with a standard deviation of 50 bytes
    "num_connections" : np.random.normal(5,2,1000),# Average 5 connections with a standard deviation of 2 connections
    "label" : ["normal"] *1000
}
# now a library to simulate attack data in a connection
attack_data = {
    "duration" : np.random.normal(40,15,200),# much shorter duration and larger packet size for attack data
    "packet_size" : np.random.normal(600,100,200),
    "num_connections" : np.random.normal(15,8,200),
    "label" : ["attack"] *200
}

#now we combine all of this into a single dataset 
#here we are creating objects of pandas dataframe for both normal and attack data
normal_df = pd.DataFrame(normal_data)
attack_df = pd.DataFrame(attack_data)
#now we are combining both dataframes into a single dataframe
full_df = pd.concat([normal_df, attack_df], ignore_index=True)

#now we are shuffling the dataset to mix normal and attack data
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# now we save it 
full_df.to_csv("network_traffic.csv", index=False)

print("Dataset created")
print(full_df.head(10)) # Display the first 10 rows of the dataset
print("\n Total rows in the dataset:", len(full_df)) # Display the total number of rows in the dataset
print("\nLabel counts:\n", full_df['label'].value_counts()) # Display the count of each label in the dataset
