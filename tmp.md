# Finetune

## Cookiecutter

````bash
pip install cookiecutter
mkdir HelloCookieCutter1
cd HelloCookieCutter1
mkdir -p {{cookiecutter.repo_name}}
cd {{cookiecutter.repo_name}}
touch {{cookiecutter.main}}.py
cd ..
touch cookiecutter.json```

```json
{
    "directory_name": "Hello",
    "file_name": "Howdy",
    "greeting_recipient": "Julie"
}```

```shell
cookiecutter /Users/gaohn/omniverse/dummy-repo/HelloCookieCutter1```
````
