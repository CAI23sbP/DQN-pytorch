from dqn import DQN
from configs import Config
import datetime

if __name__ == "__main__":
    network = DQN(Config)
    print("start_training")
    today = datetime.datetime.today()
    print(today.strftime('%Y-%m-%d %H:%M'))
    network.training()
    print("end_training")
    print(today.strftime('%Y-%m-%d %H:%M'))
    network.buffer.save()
    print("start_training")
    network.testing()
