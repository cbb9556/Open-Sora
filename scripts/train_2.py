from opensora.utils.config_utils import parse_configs


def main():
    cfg = parse_configs(training = True)
    print(cfg.get("vae"))
    print(cfg)

if __name__ == '__main__':
    main()