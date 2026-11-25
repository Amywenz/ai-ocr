import paddle
import paddlehub as hub
import paddlenlp

print(f"PaddlePaddle version: {paddle.__version__}")
print(f"PaddleHub version: {hub.__version__}")
print(f"PaddleNLP version: {paddlenlp.__version__}")

# 测试基础功能
paddle.utils.run_check()