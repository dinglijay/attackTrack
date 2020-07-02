docker build .  --build-arg SSH_PUBKEY="$(cat ~/.ssh/id_rsa.pub)" -t li-attack
docker run -ti \
        -p 6024:22 \
        -p 6025:6006 \
        --hostname attackSer2 \
        --name li-attack \
        --gpus 'all' \
        --shm-size 16g \
        -v /home/li:/workspace \
        -v /mnt/DataServer/li:/DataServer \
        li-attack