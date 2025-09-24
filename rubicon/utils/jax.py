from functools import wraps
import os


def jax_cpu_backend(func):
    """Temporarily set JAX backend to the CPU."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        original = os.environ.get("JAX_PLATFORM_NAME", "")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        try:
            return func(*args, **kwargs)
        finally:
            os.environ["JAX_PLATFORM_NAME"] = original

    return wrapper
