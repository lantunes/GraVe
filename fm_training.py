from grave import FactorizationMachine


if __name__ == '__main__':
    f = FactorizationMachine(2, 1, 1, 10)

    feature_dict = {
        "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": [], "11": [], "12": [],
        "13": [], "14": [], "15": [], "16": [], "17": [], "18": [], "19": [], "20": [], "21": [], "22": [], "23": [],
        "24": [], "25": [], "26": [], "27": [], "28": [], "29": [], "30": [], "31": [], "32": [], "33": [], "34": []
    }
    X, Y = f.build_training_data("examples/karate.walks.0", feature_dict)

    print(X)
    print(Y)
