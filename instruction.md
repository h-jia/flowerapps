# Guide: How to Publish Apps on Flower Hub

This guide walks you through the process of building, reviewing, and publishing an app on Flower Hub.

---

## 1. Build Your App

It’s showtime—build your app! 🎉

You can refer to this example app for structure and best practices:

https://flower.ai/apps/yan-gao/example-app

Please follow the **same README structure** when creating your app.

### Simulation vs. Deployment

Your app **should run smoothly in both Simulation and Deployment without code changes**. To achieve this, implement separate data loaders for each mode.

You can distinguish between Simulation and Deployment using `context.node_config` in ClientApp side. If `["partition-id", "num-partitions"]` is present, the app is running in **Simulation**.

Here’s an example:

```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the data
    batch_size = context.run_config["batch-size"]

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # **Simulation Engine**: use `flwr_datasets` and partition data on the fly
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, _ = load_sim_data(partition_id, num_partitions, batch_size)
    else:
        # **Deployment Engine**: load demo data or real user data
        data_path = context.node_config["data-path"]
        trainloader, _ = load_local_data(data_path, batch_size)

    # Training logic continues identically for both modes
```

**Recommendations:**

- For **Simulation**, use `flwr_datasets==0.6.0` for on-the-fly data partitioning: [Flower Datasets Documentation](https://flower.ai/docs/datasets/index.html)
- For **Deployment**, you may generate demo data using the CLI:
    
    `flwr-datasets create` (see the [instruction](https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html) for details).
    

---

## 2. Open a Pull Request (PR) for Review

Once your app is ready:

1. Add your project under the `flower/examples` repository.
2. Follow the [contribution guide](https://flower.ai/docs/framework/contributor-tutorial-contribute-on-github.html#creating-and-merging-a-pull-request-pr) to open a PR to the Flower main repository.

If you’re new to contributing on GitHub, you may find the [full guide](https://flower.ai/docs/framework/contributor-tutorial-contribute-on-github.html) helpful.

### Test and Format Your Code

Before opening a PR, ensure your code is properly formatted and tested within the Flower repository. First, set up a Flower development environment by following the [instructions](https://flower.ai/docs/framework/contributor-tutorial-get-started-as-a-contributor.html#create-flower-dev-environment).

Once your environment is ready, run the following commands:

```bash
./framework/dev/format.sh # Format your code
./framework/dev/test.sh   # Run tests
```

### PR Title

Use the following title format:

```
app(flowerhub): <description of your app>
```

⚠️ **Note:** This PR title will not pass CI checks. You can safely ignore this for now—this is a temporary workaround for the first batch of Flower Hub apps prior to the official launch.

Once opened, a Flower team reviewer will be notified. After review and approval:

- The PR will **not** be merged into the main repository.
- The PR will be closed after approval.
- You can proceed to publish your app on Flower Hub.

---

## 3. Create a Flower Account

Next, create a Flower account at: https://flower.ai/

Click **“Sign Up”** in the top-right corner and follow the instructions.

![Screenshot 2026-02-06 at 09.17.28.png](attachment:38537ffa-6996-4ec9-8b99-641ab9d4cbeb:Screenshot_2026-02-06_at_09.17.28.png)

### Organisation Accounts

If you’re publishing on behalf of an organisation:

- Create a regular account.
- Use your organisation name as the username (e.g., `flwrlabs` for Flower Labs).
- Update the profile with the appropriate organisation logo.

![Screenshot 2026-02-06 at 09.21.46.png](attachment:d43cff99-6631-4b1f-97a6-994d6b9a8622:Screenshot_2026-02-06_at_09.21.46.png)

⚠️ **Note:** Organisation accounts are not officially supported yet. These accounts will be migrated once organisation support is available.

---

## 4. Publish Your App on Flower Hub

Apps are published via the Flower CLI. Make sure you have installed `flwr==1.26.1`.

### Step 1: Log in to SuperGrid

```bash
flwr login supergrid
```

This command will open a browser window where you can log in using your Flower account.

![Screenshot 2026-02-06 at 09.23.37.png](attachment:3d46877c-3293-44ce-8e2d-400207a2c17d:Screenshot_2026-02-06_at_09.23.37.png)

### Step 2: Publish Your App

```bash
flwr app publish your-app-path
```

---

🎉 **Done!**

Your app is now live on Flower Hub. You can view it at:

```
https://flower.ai/apps/<account_name>/<app_name>/
```