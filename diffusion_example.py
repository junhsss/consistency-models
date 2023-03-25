from diffusers import DiffusionPipeline

repo_id = "junhsss/consistency-cifar10"
pipe = DiffusionPipeline.from_pretrained(repo_id)

pipe().image
