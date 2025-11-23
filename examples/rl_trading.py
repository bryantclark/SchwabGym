"""
Reinforcement Learning Training Example
========================================

Train a PPO agent to trade using the Schwab simulator environment.

Author: Your Name
"""

import os
import sys

# Add parent directory to path so we can import schwabgym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from schwabgym.data import load_and_clean_data, split_train_test
from schwabgym.environment import SchwabTradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_rl_agent(
    ticker: str = "AAPL",
    data_path: str = "../data/AAPL_5min.csv",
    total_timesteps: int = 100000,
    save_path: str = "./models/",
):
    """
    Train a PPO agent for algorithmic trading.

    Args:
        ticker (str): Symbol to trade
        data_path (str): Path to historical data
        total_timesteps (int): Total training steps
        save_path (str): Directory to save models
    """
    logger.info("=" * 60)
    logger.info("RL AGENT TRAINING")
    logger.info("=" * 60)

    # Resolve data path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    # Load and split data
    logger.info(f"Loading data for {ticker}...")
    df = load_and_clean_data(data_path, symbol=ticker)
    train_df, test_df = split_train_test(df, train_ratio=0.8)

    logger.info(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # Create training environment
    logger.info("Creating training environment...")

    def make_env():
        return SchwabTradingEnv(train_df, ticker=ticker, initial_cash=25000)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Create evaluation environment
    eval_env = DummyVecEnv(
        [lambda: SchwabTradingEnv(test_df, ticker=ticker, initial_cash=25000)]
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # Create callbacks
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=save_path, name_prefix="rl_model"
    )

    # Initialize PPO agent
    logger.info("Initializing PPO agent...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/",
    )

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Learning rate: 3e-4")
    logger.info(f"Batch size: 64")
    logger.info(f"Save path: {save_path}")
    logger.info("=" * 60)

    # Train
    logger.info("\nStarting training...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )

        # Save final model
        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(save_path, "vec_normalize.pkl"))

        logger.info(f"\nTraining complete! Model saved to {final_model_path}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        model.save(os.path.join(save_path, "interrupted_model"))

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 60)

    test_env = SchwabTradingEnv(test_df, ticker=ticker, initial_cash=25000)
    obs, _ = test_env.reset()

    episode_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        done = terminated or truncated

    final_nav = info.get("nav", 25000)  # Default if not in info
    initial_capital = 25000
    test_return = ((final_nav - initial_capital) / initial_capital) * 100

    logger.info(f"Test Episode Reward: {episode_reward:.4f}")
    logger.info(f"Test Return: {test_return:+.2f}%")
    logger.info(f"Final NAV: ${final_nav:,.2f}")

    # Visualize results
    # logger.info("\nGenerating performance chart...")
    # test_env.render_chart()

    return model, test_return


def load_and_evaluate(
    model_path: str, ticker: str = "AAPL", data_path: str = "../data/AAPL_5min.csv"
):
    """
    Load a trained model and evaluate it.

    Args:
        model_path (str): Path to saved model
        ticker (str): Symbol to trade
        data_path (str): Path to test data
    """
    logger.info(f"Loading model from {model_path}")

    # Resolve data path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    # Load model
    model = PPO.load(model_path)

    # Load test data
    df = load_and_clean_data(data_path, symbol=ticker)
    _, test_df = split_train_test(df, train_ratio=0.8)

    # Create environment
    env = SchwabTradingEnv(test_df, ticker=ticker, initial_cash=25000)

    # Run evaluation
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Show results
    logger.info(f"Final NAV: ${info.get('nav', 0):,.2f}")
    # env.render_chart()


def hyperparameter_search():
    """
    Search for optimal hyperparameters.

    This is a simplified example - use Optuna for more sophisticated searches.
    """
    results = []

    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [32, 64, 128]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            logger.info(f"\nTesting LR={lr}, Batch={batch_size}")

            # You would create and train a model with these hyperparameters
            # For brevity, this is left as an exercise

            # results.append({
            #     'lr': lr,
            #     'batch_size': batch_size,
            #     'test_return': test_return
            # })

    # Find best hyperparameters
    # best = max(results, key=lambda x: x['test_return'])
    # logger.info(f"\nBest hyperparameters: LR={best['lr']}, Batch={best['batch_size']}")


if __name__ == "__main__":
    # Train new agent
    model, test_return = train_rl_agent(
        ticker="AAPL",
        data_path="../data/AAPL_5min.csv",
        total_timesteps=100000,
        save_path="./models/",
    )

    # Or load existing agent
    # load_and_evaluate('./models/best_model.zip', ticker='AAPL')

    # Or run hyperparameter search
    # hyperparameter_search()
