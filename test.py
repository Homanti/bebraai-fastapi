import g4f
import requests
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import PollinationsAI

async def main():
    client = AsyncClient()

    # image = requests.get("https://images.prom.ua/2987667453_w600_h600_2987667453.jpg", stream=True).raw
    image = open("2987667453_w600_h600_2987667453_jpg.png", "rb")

    response = await client.chat.completions.create(
        model="gpt-4o",
        provider=PollinationsAI,
        messages=[
            {
                "role": "user",
                "content": "что на картинке?"
            }
        ],
        image=image,
    )

    print(response.choices[0].message.content)

asyncio.run(main())