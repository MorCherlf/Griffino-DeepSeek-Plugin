"""
DeepSeek Plugin for Griffino
监听 RabbitMQ topic，调用 DeepSeek API，返回结果
"""

import os
import json
import logging
import time
import uuid
import threading
from typing import Optional

import pika
import redis
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 环境变量 ──────────────────────────────────────────────────────────────────
RABBITMQ_HOST     = os.environ["RABBITMQ_HOST"]
RABBITMQ_PORT     = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER     = os.environ["RABBITMQ_USER"]
RABBITMQ_PASSWORD = os.environ["RABBITMQ_PASSWORD"]

REDIS_HOST     = os.environ["REDIS_HOST"]
REDIS_PORT     = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_USER     = os.environ.get("REDIS_USER", "")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

DEEPSEEK_API_KEY  = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MAX_TOKENS        = int(os.environ.get("MAX_TOKENS", "4096"))
SYSTEM_PROMPT     = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
PLUGIN_ID         = os.environ.get("PLUGIN_ID", "cc.griffino.deepseek")

# Topic 常量
REQUEST_TOPIC  = "invoke.cc.griffino.deepseek.ai_chat_model.v1"
EXCHANGE_NAME  = "griffino.plugins"

# ── Redis 用户配置读取 ─────────────────────────────────────────────────────────
def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USER or None,
        password=REDIS_PASSWORD or None,
        decode_responses=True,
    )

def get_user_config(rdb: redis.Redis, user_id: str) -> dict:
    """读取用户个人配置，返回 dict，不存在则返回空 dict"""
    key = f"user:{user_id}:plugin:{PLUGIN_ID}:config"
    try:
        raw = rdb.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        log.warning(f"读取用户配置失败 user_id={user_id}: {e}")
    return {}

# ── DeepSeek 调用 ─────────────────────────────────────────────────────────────
def call_deepseek(messages: list, user_config: dict, overrides: dict) -> dict:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    # 合并 system prompt：全局 + 用户个人
    system_parts = [SYSTEM_PROMPT]
    user_system_prompt = user_config.get("USER_SYSTEM_PROMPT", "").strip()
    if user_system_prompt:
        system_parts.append(user_system_prompt)

    # 处理回复语言
    reply_language = user_config.get("REPLY_LANGUAGE", "auto")
    if reply_language == "zh-CN":
        system_parts.append("请始终使用简体中文回复。")
    elif reply_language == "en-US":
        system_parts.append("Always reply in English.")

    full_system_prompt = "\n\n".join(system_parts)

    # 构建消息列表：system 消息放最前
    final_messages = [{"role": "system", "content": full_system_prompt}]
    # 过滤掉原消息里的 system 角色（防止重复），保留 user/assistant
    for msg in messages:
        if msg.get("role") != "system":
            final_messages.append(msg)

    # 参数优先级：overrides > 用户配置 > 全局默认
    temperature = float(
        overrides.get("temperature")
        or user_config.get("TEMPERATURE")
        or 1.0
    )
    max_tokens = int(
        overrides.get("maxTokens")
        or MAX_TOKENS
    )
    model = DEEPSEEK_MODEL

    log.info(f"调用 DeepSeek model={model} temperature={temperature} max_tokens={max_tokens} messages={len(final_messages)}")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        usage   = resp.usage
        return {
            "ok":      True,
            "content": content,
            "model":   model,
            "usage": {
                "promptTokens":     usage.prompt_tokens,
                "completionTokens": usage.completion_tokens,
                "totalTokens":      usage.total_tokens,
            },
        }
    except Exception as e:
        log.error(f"DeepSeek API 调用失败: {e}")
        return {"ok": False, "error": str(e)}

# ── RabbitMQ 消息处理 ─────────────────────────────────────────────────────────
def process_message(ch, method, properties, body, rdb: redis.Redis):
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        log.error(f"消息解析失败: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    user_id         = payload.get("userId", "")
    messages        = payload.get("messages", [])
    overrides       = payload.get("overrides", {})
    correlation_id  = properties.correlation_id or str(uuid.uuid4())
    reply_to        = properties.reply_to

    log.info(f"收到请求 user_id={user_id} correlation_id={correlation_id} messages={len(messages)}")

    # 读取用户个人配置
    user_config = get_user_config(rdb, user_id) if user_id else {}

    # 调用 DeepSeek
    result = call_deepseek(messages, user_config, overrides)

    # 发送响应
    if reply_to:
        try:
            ch.basic_publish(
                exchange="",
                routing_key=reply_to,
                properties=pika.BasicProperties(
                    correlation_id=correlation_id,
                    content_type="application/json",
                ),
                body=json.dumps(result, ensure_ascii=False),
            )
            log.info(f"响应已发送 correlation_id={correlation_id} ok={result.get('ok')}")
        except Exception as e:
            log.error(f"发送响应失败: {e}")
    else:
        log.warning(f"消息无 reply_to，丢弃响应 correlation_id={correlation_id}")

    ch.basic_ack(delivery_tag=method.delivery_tag)

# ── 连接 RabbitMQ（含重试）────────────────────────────────────────────────────
def connect_rabbitmq(max_retries: int = 10, retry_interval: int = 5) -> pika.BlockingConnection:
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
    )
    for attempt in range(1, max_retries + 1):
        try:
            conn = pika.BlockingConnection(params)
            log.info(f"RabbitMQ 连接成功 ({RABBITMQ_HOST}:{RABBITMQ_PORT})")
            return conn
        except Exception as e:
            log.warning(f"RabbitMQ 连接失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_interval)
    raise RuntimeError("RabbitMQ 连接失败，已超过最大重试次数")

# ── 主循环 ────────────────────────────────────────────────────────────────────
def main():
    log.info(f"DeepSeek Plugin 启动 plugin_id={PLUGIN_ID}")
    log.info(f"DeepSeek 配置: base_url={DEEPSEEK_BASE_URL} model={DEEPSEEK_MODEL} max_tokens={MAX_TOKENS}")

    # 初始化 Redis 客户端
    rdb = get_redis_client()
    try:
        rdb.ping()
        log.info(f"Redis 连接成功 ({REDIS_HOST}:{REDIS_PORT})")
    except Exception as e:
        log.error(f"Redis 连接失败: {e}")
        raise

    # 连接 RabbitMQ
    conn = connect_rabbitmq()
    ch   = conn.channel()

    # 使用和 provisioner 一致的队列名，直接消费，不重复声明 exchange
    queue_name = f"plugin.{PLUGIN_ID}.ai_chat_model"
    # passive=True 表示只检查队列是否存在，不重新声明（已由 Griffino 声明好）
    ch.queue_declare(queue=queue_name, durable=True, passive=True)

    # 限制每次只处理一条消息（避免并发导致 DeepSeek 限流）
    ch.basic_qos(prefetch_count=4)

    log.info(f"开始监听 exchange={EXCHANGE_NAME} topic={REQUEST_TOPIC}")

    def on_message(ch, method, properties, body):
        process_message(ch, method, properties, body, rdb)

    ch.basic_consume(queue=queue_name, on_message_callback=on_message)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        log.info("收到退出信号，正在关闭...")
        ch.stop_consuming()
    finally:
        conn.close()
        log.info("DeepSeek Plugin 已退出")


if __name__ == "__main__":
    main()
