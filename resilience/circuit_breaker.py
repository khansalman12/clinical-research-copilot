import time
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL_FAST

FAILURE_THRESHOLD = 3
RECOVERY_TIMEOUT = 60
LLM_CALL_TIMEOUT = 30


class CircuitBreaker:
    """
    CLOSED -> normal operation.
    OPEN -> 3 consecutive failures tripped it; calls route straight to the
      fallback model without hitting the primary.
    HALF-OPEN -> recovery timeout elapsed; the next call tests the primary again.
    """

    def __init__(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._fallback_client = Groq(api_key=GROQ_API_KEY)

    def _is_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= RECOVERY_TIMEOUT:
                self.state = "HALF-OPEN"
                print("[CircuitBreaker] → HALF-OPEN: testing recovery...")
                return False
            return True
        return False

    def _record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= FAILURE_THRESHOLD:
            self.state = "OPEN"
            print(f"[CircuitBreaker] → OPEN after {self.failure_count} consecutive failures.")

    def call(self, client: Groq, model: str, messages: list, **kwargs) -> str:
        if self._is_open():
            print(f"[CircuitBreaker] OPEN — routing to fallback model: {LLM_MODEL_FAST}")
            response = self._fallback_client.chat.completions.create(
                model=LLM_MODEL_FAST,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=LLM_CALL_TIMEOUT,
                **kwargs,
            )
            self._record_success()
            return response.choices[0].message.content

        except Exception as e:
            print(f"[CircuitBreaker] Call failed: {e}")
            self._record_failure()
            print(f"[CircuitBreaker] Falling back to: {LLM_MODEL_FAST}")
            response = self._fallback_client.chat.completions.create(
                model=LLM_MODEL_FAST,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content


breaker = CircuitBreaker()
