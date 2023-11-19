import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from agents.portfolio_rebalancer import PortfolioRebalancer
from envs.market import StockMarket
from value_functions.sarsa import LinearSarsa

MAX_MEMORY = "32g"
spark = (
    SparkSession.builder.appName("Learning Shell")
    .config("spark.driver.memory", MAX_MEMORY)
    .config("spark.sql.session.timeZone", "Asia/Seoul")
    .config("spark.driver.extraJavaOptions", "-Duser.timezone=Asia/Seoul")
    .config("spark.executor.extraJavaOptions", "-Duser.timezone=Asia/Seoul")
    .getOrCreate()
)

DATA_PATH = "./data"

csv_paths = [
    "Celltrion.csv",
    "HyundaiMotor.csv",
    "Kakao.csv",
    "KOSPI.csv",
    "LGChemical.csv",
    "LGH&H.csv",
    "NAVER.csv",
    "SamsungBiologics.csv",
    "SamsungElectronics.csv",
    "SamsungElectronics2.csv",
    "SamsungSDI.csv",
    "SKhynix.csv",
]

if __name__ == "__main__":
    datamart = {}
    for stock, csv_path in zip(StockMarket.stock_list, csv_paths):
        datamart[stock] = spark.read.csv(
            os.path.join(DATA_PATH, csv_path), header=True
        ).cache()

        if stock == "kospi":

            @udf(FloatType())
            def floatize(volume):
                if volume.endswith("K"):
                    volume = volume.replace("K", "000")
                return float(volume.replace(",", "").replace(" ", ""))

            datamart[stock] = datamart[stock].withColumn("Volume", floatize("Volume"))

    env_train = StockMarket(
        datamart=datamart,
        start_date=datetime.date(2016, 1, 1),
        end_date=datetime.date(2019, 12, 31),
        episode_length=10,
    )

    linear_sarsa = LinearSarsa(
        dim_state=517,
        dim_action=13,
        learning_rate=0.1,
        discount_factor=1,
    )

    agent = PortfolioRebalancer(Q=linear_sarsa, epsilon=0.1)

    epoch = 10000
    for i_episode in range(epoch):
        observation = env_train.emit_observation(timestep=0)
        _return = 0
        for _ in range(10):
            state = agent.struct_state(observation=observation)
            action = agent.act(state=state)
            observation, reward, done, info = env_train.step(
                action=agent.to_market_action(state=state, action=action),
                portfolio=agent.portfolio,
            )
            agent.Q.update(
                reward=reward,
                curr_state=state,
                curr_action=action,
                next_state=agent.struct_state(observation=observation),
            )
            _return += reward
            if done:
                print("Episode finished after {} timesteps".format(env_train.t + 1))
                print("ROR: {:.2%}\n".format(_return / 100_000_000))
                env_train.reset()
                agent.reset()
                break

    env_val = StockMarket(
        datamart=datamart,
        start_date=datetime.date(2021, 1, 1),
        end_date=datetime.date(2021, 5, 19),
        episode_length=10,
    )

    n = len(env_val.start_point_list)
    total_return = 0
    for _ in range(n):
        observation = env_val.emit_observation(timestep=0)
        _return = 0
        for _ in range(10):
            state = agent.struct_state(observation=observation)
            action = agent.act(state=state)
            observation, reward, done, info = env_val.step(
                action=action, portfolio=agent.portfolio
            )
            _return += reward
            if done:
                print("ROR: {:.2%}\n".format(_return / 100_000_000))
                total_return += _return
                env_val.reset()
                agent.reset()
                break

    print("Final Average ROR: {:.2%}".format(total_return / n))
