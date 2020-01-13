require 'sinatra'
require 'json'


helpers do
  def json_or_default?(type)
    ['application/json', 'application/*', '*/*'].include?(type.to_s)
  end

  def accepted_media_type
    return 'json' unless request.accept.any?

    request.accept.each do |mt|
      return 'json' if json_or_default?(mt)
    end

    # not acceptable format
    content_type 'text/plain'
    halt 406, 'application/json, application/xml'
  end

  def type
    @type ||= accepted_media_type
  end

  def send_data(data = {})
    if type == 'json'
      content_type 'application/json'
      data[:json].call.to_json if data[:json]
      # other accepted media types go below here
    end
  end

  def model_eval(title, content)
    model_module = ENV['MODEL_MODULE']
    cmd = "python3 #{model_module} eval '#{title}' '#{content}'"
    output = `#{cmd}`
    puts("Received output on stdout:\n #{output}")
    lines = output.split("\n")
    puts(lines)
    raise "Couldn't run prediction" if lines.empty?

    output.split("\n")[-1]
  end
end

post '/predict_publication' do
  halt 415 unless request.env['CONTENT_TYPE'] == 'application/json'

  begin
    example = JSON.parse(request.body.read)
    pred_pub = model_eval(example['title'], example['content'])
  rescue JSON::ParserError => e
    halt 400, send_data(json: -> { { message: e. to_s } })
  end

  send_data(json: -> { { publication: pred_pub }})

end
