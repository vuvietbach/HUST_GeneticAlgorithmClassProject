def main():
    dataset = ["cogent_center_0", 
               "cogent_center_1",
               "cogent_center_2",
                "cogent_center_3",
               "cogent_center_4",
               "cogent_rural_0", 
                "cogent_rural_1",
                "cogent_rural_2",
                "cogent_rural_3",
                "cogent_rural_4",
               "cogent_uniform_0",
               "cogent_uniform_1"]
    root_dir = './dataset'
    cmd = 'python vnf.py --input {} --request {}'
    for d in dataset:
        dataset_dir = root_dir + '/' + d
        input_file = dataset_dir + '/input.txt'
        request_files = [dataset_dir + '/request10.txt', dataset_dir + '/request20.txt', dataset_dir + '/request30.txt']
        for r in request_files:
            with open("script.sh", "a") as f:
                f.write(cmd.format(input_file, r) + '\n')

if __name__ == "__main__":
    main()