from prefect import flow

@flow(log_prints=True)
def mastering_flow() -> str:
    print("Prefect flow is running")
