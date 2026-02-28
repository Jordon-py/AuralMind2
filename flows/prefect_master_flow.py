from prefect import flow

@flow(log_prints=True)
def mastering_flow():
    print("Prefect flow is running")
