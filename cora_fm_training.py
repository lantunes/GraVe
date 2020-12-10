from grave import FactorizationMachine

if __name__ == '__main__':

    X, Y, dictionary = FactorizationMachine.load_training_data("out/cora.fm.training.bitwise_or.sparse.data", sparse=True)

    fm = FactorizationMachine(dim=128, y_max=100, alpha=0.75, context_window_size=10, dictionary=dictionary)

    fm.fit(X, Y, batch_size=1, num_epochs=50, learning_rate=0.002, use_autograd=False)

    fm.save("cora.fm.model")