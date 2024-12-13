from mymodule.helper import greet
from mynats.sync_nats import Nats


def example():
    print(greet("Ivo"))
    try:
        print("Testing NATS")
        store = Nats()
        key = "foo"
        value = "bar"
        print(f'put {key} -> value {value}')
        store.put(key, value)
        val = store.get(key)
        print(f'get {key} <- value {val}')
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    example()
