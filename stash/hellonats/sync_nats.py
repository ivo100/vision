import asyncio
import atexit

import nats
import concurrent.futures

class Nats:
    def __init__(self, connection_url="nats://mini:4222", bucket="TESTKV"):
        """
        Initialize a synchronous NATS Key-Value wrapper.

        :param connection_url: Optional NATS connection URL
        """
        self.bucket = bucket
        self._connection_url = connection_url
        self._loop = asyncio.get_event_loop()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._nc = self.connect()
        self._js = self._nc.jetstream()
        self._kv = self.create_store(bucket)
        atexit.register(self._cleanup)


    def _cleanup(self):
        self.close()

    def _run_async(self, coro):
        """
        Run an async coroutine in the current event loop.

        :param coro: Coroutine to run
        :return: Result of the coroutine
        """
        try:
            return self._loop.run_until_complete(coro)
        except Exception as e:
            print(f"Error executing coroutine: {e}")
            raise

    def connect(self):
        return self._run_async(nats.connect(self._connection_url))

    def create_store(self, bucket: str):
        """
        Create a Key-Value store synchronously.

        :param bucket:
        :param nc: NATS connection
        :return: Key-Value store
        """
        if len(bucket) > 0:
            self.bucket = bucket
        return self._run_async(self._js.create_key_value(bucket=self.bucket))

    def put(self, key:str, value: str):
        """
        Put a key-value pair synchronously.

        :param kv_store: Key-Value store
        :param key: Key to store
        :param value: Value to store
        """
        return self._run_async(self._kv.put(key, bytes(value, 'utf-8')))

    def get(self, key:str)->str:
        """
        Retrieve a value synchronously.

        :param kv_store: Key-Value store
        :param key: Key to retrieve
        :return: Key-Value entry
        """
        kv = self._run_async(self._kv.get(key))
        if kv is None:
            return None
        return str(kv.value, 'utf-8')

    def close(self):
        try:
            if self._nc is not None:
                self._run_async(self._nc.close())
        except Exception as e:
            print(f"Error closing connection: {e}")
        finally:
            # Cleanup executor
            self._executor.shutdown(wait=False)

def example():
    try:
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

