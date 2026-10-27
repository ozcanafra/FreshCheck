
def file_size_of(path_text):
    try:
        return Path(path_text).stat().st_size
    except OSError:
        return 0


def keeper_key(p):
    # 1. En yüksek çözünürlük (genişlik x yükseklik)
    # 2. Çözünürlük eşitse en büyük dosya boyutu
    # 3. Eşitse augmented olmayan görüntü
    # 4. Eşitse test > valid > train
    # 5. Eşitse daha çok kopya bağlantısı
    # 6. Son eşitlikte dosya yolu
    return (
        -pixels_of(p),
        -file_size_of(p),
        is_aug(p),
        SPLIT_PRIORITY.get(split_of(p), 3),
        -len(graph[p]),
        p.lower(),
    )
