from roboflow import Roboflow

rf = Roboflow(api_key="kNAPfqhoKBITl8TzWEoR")
project = rf.workspace().project("sheep_farming")
model = project.version("2").model

job_id, signed_url, expire_time = model.predict_video(
    "D:\pi515\sheeps_5min.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)
