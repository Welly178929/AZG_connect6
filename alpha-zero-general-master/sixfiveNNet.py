import os
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet  # 若無可用空的 class 替代
from .connect6_5.keras.sixfiveNet import GomokuAndSixNNet as onnet  # 你的模型類別路徑請對應
from utils import dotdict
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ 全域訓練參數
args = dotdict({
    'lr': 0.002,
    'dropout': 0.3,
    'epochs': 50,
    'batch_size': 256,
    'cuda': True,
    'num_channels': 128,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples, examples_filename=None):
        """
        Train the network on provided examples and plot training loss curves.

        Args:
          examples: list of (board, pi, v)
          examples_filename: path to the .examples file, used to derive the plot filename
        Returns:
          history: Keras History object
        """
        # 準備資料
        input_boards, target_pis, target_vs = zip(*examples)
        input_boards = np.asarray(input_boards)
        target_pis   = np.asarray(target_pis)
        target_vs    = np.asarray(target_vs)

        # 設定 callbacks：EarlyStopping 和 ReduceLROnPlateau
        early_stop = EarlyStopping(
            monitor='pi_loss', patience=10,
            restore_best_weights=True, verbose=1
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='pi_loss', factor=0.5,
            patience=5, verbose=1
        )

        # 訓練模型並取得 history
        history = self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )

        # 決定儲存路徑
        if examples_filename:
            base = os.path.basename(examples_filename)
            name = os.path.splitext(base)[0]
            save_path = os.path.join(os.path.dirname(examples_filename), f"{name}.png")
        else:
            save_path = 'training_loss.png'

        # 繪製並儲存 loss 曲線
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get('pi_loss', []), label='Policy Loss')
        plt.plot(history.history.get('v_loss', []), label='Value Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"✅ Saved training loss plot to {save_path}")
        return history

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board, verbose=False)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.nnet.model.save_weights(filepath)
        print(f"✅ Model saved to {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ No model found at {filepath}")
        self.nnet.model.load_weights(filepath)
        print(f"📥 Model loaded from {filepath}")