import requests
import random
import matplotlib.pyplot as plt
import pandas as pd


#
# # collect data by users API
# # id_ = 0
# # response = requests.get('https://api.github.com/users?since=' + str(id_), headers=headers,)
# # data = response.json()
#
# # collect data by search API
# # response = requests.get('https://api.github.com/search/users?q=created:<2021-02-22&created:>2021-02-22', headers=headers)
#

headers = {
    'Authorization': 'token 72852dab0e847c725be2942974d1b7652b4cc2c3',  # replace <TOKEN> with your token
}

# Get Validation Estimate
def validation_process():
    count = 0
    current_id = 1
    max_id = 1000
    while current_id < max_id:
        response = requests.get('https://api.github.com/users?since=' + str(current_id) + "?per_page=100", headers=headers)
        data = response.json()
        current_id = data[-1]['id']
        print(current_id)
        if current_id > max_id:
            for i in range(len(data)):
                if data[i]['id'] > max_id:
                    current_id = data[i - 1]['id']
                    data = data[:i].copy()

                    break
        count += len(data)
        print("Progress: " + str(round(current_id / max_id, 4) * 100) + " %")

    print("Out of the first: " + str(max_id) + " users,")
    print("there are: " + str(count) + " active users!")


# UNCOMMENT TO RUN VALIDATION PROCESS
# validation_process()


#Sample function
def sample(sample_ids, bucket_size):
    sample_densities = []
    for id in sample_ids:
        starting_id = id * bucket_size
        count = 0
        max_id = starting_id + bucket_size
        while starting_id < max_id:
            data = requests.get('https://api.github.com/users?since=' + str(starting_id), headers=headers).json()
            starting_id = data[-1]['id']
            if starting_id > max_id:
                for i in range(len(data)):
                    if data[i]['id'] > max_id:
                        data = data[:i].copy()
                        break
            count += len(data)

        sample_densities.append(count / bucket_size)
    sample_data = sum(sample_densities) / len(sample_densities)
    return sample_densities

#Estimation function
def estimate(sample_data, MAXID):
    estimation = [x * MAXID for x in sample_data]
    return estimation


# token = "72852dab0e847c725be2942974d1b7652b4cc2c3"
# max_id = 79471337

headers = {
    'Authorization': 'token 72852dab0e847c725be2942974d1b7652b4cc2c3',  # replace <TOKEN> with your token
}
print("Github Active User Estimation")
num_samples = [10, 20, 30, 40, 50, 100, 200]
bucket_size = 90
MAXID = 79471337
num_buckets = MAXID // bucket_size

df = pd.DataFrame()
for num in num_samples:
    est_list = []
    for i in range(10):
        sample_ids = random.sample(range(num_buckets), num)
        sample_densities = sample(sample_ids, bucket_size)
        est = estimate(sample_densities, MAXID)
        est_list.append(sum(est) / len(est))
    df1 = pd.DataFrame({str(num): est_list})
    df = pd.concat([df, df1], axis=1)
    df.to_csv(('GithubSamples' + str(num)))
    print("Finished " + str(num) + " samples! ")

# PLOT RESULTS
bplot = df.boxplot()
bplot.set_ylabel('Active Github User Estimation')
bplot.set_xlabel('Number of Samples')
plt.savefig('GithubFull1')
plt.show()


