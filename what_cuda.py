import torch

print('활성화?',torch.cuda.is_available())
print('장치 이름:', torch.cuda.get_device_properties('cuda').name)
print('FlashAttention 사용 가능:', torch.backends.cuda.flash_sdp_enabled())
print(f'torch 버전: {torch.__version__ } ')

x = torch.rand(5, 3)
print(x)