FROM python:3.11-slim

USER root

ENV PYTHONUNBUFFERED=1
WORKDIR /agent

RUN apt-get update \
    && apt-get install -y --no-install-recommends iptables \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyyaml

COPY environment/questions.txt /agent/questions.txt
COPY environment/format_guide.md /agent/format_guide.md
COPY environment/deployment_notes.txt /agent/deployment_notes.txt
COPY environment/notes_to_self.txt /opt/swarm_seed/.notes_to_self.txt
COPY swarm /opt/swarm
COPY swarm_eval_agent.py /opt/agent.py

RUN useradd -m -s /bin/bash dev \
    && chown -R dev:dev /agent \
    && chmod 600 /opt/swarm_seed/.notes_to_self.txt \
    && chmod 700 /opt/agent.py \
    && chmod 700 /opt \
    && chmod -R go-rwx /opt

ENTRYPOINT ["python", "/opt/agent.py"]
