import asyncio
import aiohttp
import base64


async def test():
    with open("test.wav", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
            headers={
                "Authorization": "oMowmlxsAdR3TqmLaY7mG_SBYTjGmg_p124n8G6FKA67dkmoDLWBtNElIpQQqxHk",
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            json={
                "pipelineTasks": [{
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": "hi"},
                        "serviceId": "bhashini/ai4bharat/conformer-multilingual-asr",
                    }
                }],
                "inputData": {
                    "audio": [{"audioContent": audio_b64}]
                }
            }
        ) as resp:
            print(resp.status)
            print(await resp.json())


asyncio.run(test())