html:
	jupyter-book build yet_another_jax_transformer

clean:
	rm -r yet_another_jax_transformer/_build

publish: html
	ghp-import -n -p -f yet_another_jax_transformer/_build/html
